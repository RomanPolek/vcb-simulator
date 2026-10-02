/* Event-driven simulation: changed nets schedule readers for the next tick. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdint.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#define API __declspec(dllexport)
#else
#include <time.h>
#define API __attribute__((visibility("default")))
#endif

typedef struct {
    uint32_t gate_id, input_start, input_count, output_start, output_count;
    uint8_t kind, value_bit, initial_state;
} NativeOp;

#include "native_plan.h"

static uint8_t gate_out[NATIVE_GATE_COUNT + 1];
static int32_t net_counts[NATIVE_NET_COUNT + 1];
static int32_t next_net_counts[NATIVE_NET_COUNT + 1];
static uint8_t changed_flags[NATIVE_NET_COUNT + 1];
static uint32_t changed_ids[NATIVE_NET_COUNT + 1];
static uint32_t changed_count;
static uint8_t input_prev[NATIVE_INPUT_COUNT + 1];
static uint32_t rng_state[NATIVE_GATE_COUNT + 1];
static uint32_t work_a[NATIVE_OP_COUNT + 1], work_b[NATIVE_OP_COUNT + 1];
static uint32_t *work = work_a, *next_work = work_b;
static uint32_t work_count, next_work_count;
static uint8_t next_flags[NATIVE_OP_COUNT + 1];
static uint32_t vmem_words[NATIVE_VMEM_COUNT];
static uint32_t vmem_address_value, vmem_loaded_address, vmem_data_value;
static uint8_t vmem_lock_pending, vmem_deferred_publish;
static uint8_t vmem_locked_this_tick, vmem_address_changed, vmem_data_changed;
static uint32_t clock_value, clock_ticks_until_toggle;
static uint64_t tick_count;

static uint32_t xorshift32(uint32_t value) {
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
}

static void commit_net(uint32_t net, int32_t delta) {
    if (net == 0 || net > NATIVE_NET_COUNT) return;
    next_net_counts[net] += delta;
    if (!changed_flags[net]) {
        changed_flags[net] = 1;
        changed_ids[changed_count++] = net;
    }
}

static void set_gate(const NativeOp *op, uint8_t value) {
    uint32_t gate = op->gate_id;
    uint8_t old = gate_out[gate];
    if (old == value) return;
    gate_out[gate] = value;
    int32_t delta = (int32_t)value - (int32_t)old;
    for (uint32_t i = 0; i < op->output_count; ++i)
        commit_net(native_outputs[op->output_start + i], delta);
}

static void seed_gate(const NativeOp *op, uint8_t value) {
    gate_out[op->gate_id] = value;
    if (!value) return;
    for (uint32_t i = 0; i < op->output_count; ++i) {
        uint32_t net = native_outputs[op->output_start + i];
        if (net > 0 && net <= NATIVE_NET_COUNT) {
            ++net_counts[net];
            ++next_net_counts[net];
        }
    }
}

static uint8_t net_on(uint32_t net) {
    return net > 0 && net <= NATIVE_NET_COUNT && net_counts[net] != 0;
}

static void evaluate(uint32_t op_id) {
    const NativeOp *op = &native_ops[op_id];
    uint8_t value = 0;
    uint32_t n = op->input_count, start = op->input_start;
    switch (op->kind) {
    case 2: case 5: case 7: case 8: /* OR-like */
        for (uint32_t i = 0; i < n; ++i)
            if (net_on(native_inputs[start + i])) { value = 1; break; }
        break;
    case 4: case 6: /* NOT-like */
        value = 1;
        for (uint32_t i = 0; i < n; ++i)
            if (net_on(native_inputs[start + i])) { value = 0; break; }
        break;
    case 1: /* AND */
        value = n != 0;
        for (uint32_t i = 0; i < n; ++i)
            if (!net_on(native_inputs[start + i])) { value = 0; break; }
        break;
    case 3: /* XOR */
        for (uint32_t i = 0; i < n; ++i)
            value ^= net_on(native_inputs[start + i]);
        break;
    case 10: { /* RANDOM: advance even while disabled */
        uint32_t state = xorshift32(rng_state[op->gate_id]);
        rng_state[op->gate_id] = state;
        uint8_t enabled = n == 0;
        for (uint32_t i = 0; i < n; ++i)
            if (net_on(native_inputs[start + i])) { enabled = 1; break; }
        value = enabled ? (uint8_t)(state & 1u) : 0;
        break;
    }
    case 9: case 11: case 12: { /* rising-edge storage */
        uint32_t rising = 0;
        for (uint32_t i = 0; i < n; ++i) {
            uint8_t current = net_on(native_inputs[start + i]);
            rising += current && !input_prev[start + i];
            input_prev[start + i] = current;
        }
        if (op->kind == 9) {
            if (!rising) return;
            value = gate_out[op->gate_id] ^ 1u;
        } else {
            if (!(rising & 1u) || vmem_locked_this_tick) return;
            uint32_t bit = 1u << (op->value_bit == 255 ? 0 : op->value_bit);
            if (op->kind == 11) {
                vmem_address_value ^= bit;
                vmem_address_changed = 1;
                value = (vmem_address_value & bit) != 0;
            } else {
                vmem_data_value ^= bit;
                vmem_data_changed = 1;
                return;
            }
        }
        break;
    }
    default:
        return;
    }
    set_gate(op, value);
}

static void step_clock(void) {
    if (!NATIVE_CLOCK_COUNT) return;
    if (clock_ticks_until_toggle) {
        --clock_ticks_until_toggle;
        return;
    }
    clock_value ^= 1u;
    clock_ticks_until_toggle = NATIVE_CLOCK_INTERVAL - 1u;
    for (uint32_t i = 0; i < NATIVE_CLOCK_COUNT; ++i)
        set_gate(&native_clocks[i], (uint8_t)clock_value);
}

static void publish_deferred(void) {
    if (!vmem_deferred_publish) return;
    for (uint32_t i = 0; i < NATIVE_VMEM_DATA_COUNT; ++i) {
        const NativeOp *op = &native_ops[native_vmem_data_ops[i]];
        uint8_t value = op->value_bit < 32 ?
            (uint8_t)((vmem_data_value >> op->value_bit) & 1u) : 0;
        set_gate(op, value);
    }
    vmem_deferred_publish = 0;
}

static void finalize_vmem(void) {
    if (!NATIVE_VMEM_ENABLED) return;
    if (vmem_data_changed) {
        uint32_t old = vmem_words[vmem_loaded_address];
        vmem_words[vmem_loaded_address] =
            (old & ~NATIVE_VMEM_DATA_MASK) |
            (vmem_data_value & NATIVE_VMEM_DATA_MASK);
        vmem_deferred_publish = 1;
    }
    if (vmem_locked_this_tick) {
        vmem_lock_pending = 0;
    } else if (vmem_address_changed) {
        vmem_loaded_address = vmem_address_value & NATIVE_VMEM_ADDRESS_MASK;
        vmem_data_value = vmem_words[vmem_loaded_address] & NATIVE_VMEM_DATA_MASK;
        vmem_deferred_publish = 1;
        vmem_lock_pending = 1;
    }
}

static void publish_changed_nets(void) {
    for (uint32_t i = 0; i < changed_count; ++i) {
        uint32_t net = changed_ids[i];
        uint8_t old_on = net_counts[net] != 0;
        uint8_t new_on = next_net_counts[net] != 0;
        net_counts[net] = next_net_counts[net];
        changed_flags[net] = 0;
        if (old_on == new_on) continue;
        for (uint32_t j = native_reader_offsets[net];
             j < native_reader_offsets[net + 1]; ++j) {
            uint32_t op = native_reader_ops[j];
            if (next_flags[op]) continue;
            next_flags[op] = 1;
            next_work[next_work_count++] = op;
        }
    }
}

API void native_load_memory(const uint32_t *words, uint32_t count) {
    memset(vmem_words, 0, sizeof(vmem_words));
    if (words && count) {
        if (count > NATIVE_VMEM_COUNT) count = NATIVE_VMEM_COUNT;
        memcpy(vmem_words, words, count * sizeof(uint32_t));
    }
}

API void native_reset_state(void) {
    memset(gate_out, 0, sizeof(gate_out));
    memset(net_counts, 0, sizeof(net_counts));
    memset(next_net_counts, 0, sizeof(next_net_counts));
    memset(changed_flags, 0, sizeof(changed_flags));
    memset(input_prev, 0, sizeof(input_prev));
    memset(next_flags, 0, sizeof(next_flags));
    changed_count = next_work_count = 0;
    clock_value = clock_ticks_until_toggle = 0;
    vmem_address_value = vmem_loaded_address = 0;
    vmem_data_value = NATIVE_VMEM_ENABLED ?
        (vmem_words[0] & NATIVE_VMEM_DATA_MASK) : 0;
    vmem_lock_pending = vmem_deferred_publish = 0;
    vmem_locked_this_tick = vmem_address_changed = vmem_data_changed = 0;
    tick_count = 0;
    work = work_a;
    next_work = work_b;
    for (uint32_t gate = 0; gate <= NATIVE_GATE_COUNT; ++gate) {
        uint32_t seed = 0x9e3779b9u ^ gate;
        rng_state[gate] = seed ? seed : 1u;
    }
    for (uint32_t i = 0; i < NATIVE_OP_COUNT; ++i) {
        const NativeOp *op = &native_ops[i];
        if (op->kind == 9)
            seed_gate(op, op->initial_state);
    }
    if (NATIVE_VMEM_ENABLED) {
        for (uint32_t i = 0; i < NATIVE_OP_COUNT; ++i) {
            const NativeOp *op = &native_ops[i];
            if (op->kind == 11 || op->kind == 12) {
                uint32_t word = op->kind == 11 ?
                    vmem_address_value : vmem_data_value;
                uint8_t value = op->value_bit < 32 ?
                    (uint8_t)((word >> op->value_bit) & 1u) : 0;
                seed_gate(op, value);
            }
        }
    }
    work_count = NATIVE_INITIAL_WORK_COUNT;
    memcpy(work, native_initial_work, work_count * sizeof(uint32_t));
}

API void native_tick(void) {
    changed_count = 0;
    next_work_count = 0;
    for (uint32_t i = 0; i < work_count; ++i) next_flags[work[i]] = 0;
    vmem_locked_this_tick = vmem_lock_pending;
    vmem_address_changed = vmem_data_changed = 0;
    publish_deferred();
    step_clock();
    for (uint32_t i = 0; i < NATIVE_RANDOM_COUNT; ++i)
        evaluate(native_random_ops[i]);
    for (uint32_t i = 0; i < work_count; ++i)
        evaluate(work[i]);
    finalize_vmem();
    publish_changed_nets();
    uint32_t *old = work;
    work = next_work;
    next_work = old;
    work_count = next_work_count;
    ++tick_count;
}

API void native_tick_many(uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) native_tick();
}

/* Stay in C between snapshots, so rendering cannot stall every tick batch
 * while Python reacquires its interpreter lock. Poll time about every 0.25 ms;
 * adapt the poll interval to the circuit rather than assuming a fixed TPS. */
static uint64_t wall_counter(void) {
#ifdef _WIN32
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);
    return (uint64_t)value.QuadPart;
#else
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t)value.tv_sec * 1000000000ull + (uint64_t)value.tv_nsec;
#endif
}

API void native_tick_for(double seconds) {
    if (!(seconds > 0.0)) return;
#ifdef _WIN32
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    uint64_t hz = (uint64_t)frequency.QuadPart;
#else
    uint64_t hz = 1000000000ull;
#endif
    uint64_t previous = wall_counter();
    uint64_t deadline = previous + (uint64_t)(seconds * (double)hz);
    uint64_t target = hz / 4000u;
    uint32_t batch = 64u;
    while (previous < deadline) {
        native_tick_many(batch);
        uint64_t now = wall_counter();
        uint64_t elapsed = now - previous;
        if (now >= deadline) break;
        if (elapsed) {
            uint64_t remaining = deadline - now;
            uint64_t desired = remaining < target ? remaining : target;
            uint64_t next = (uint64_t)batch * desired / elapsed;
            /* Limit growth after a short sample; never use circuit metadata. */
            if (next > (uint64_t)batch * 4u) next = (uint64_t)batch * 4u;
            if (next > 65536u) next = 65536u;
            batch = next ? (uint32_t)next : 1u;
        }
        previous = now;
    }
}

API const uint8_t *native_gate_out(void) { return gate_out; }
API const int32_t *native_net_counts(void) { return net_counts; }
API const int32_t *native_next_net_counts(void) { return next_net_counts; }
API const uint32_t *native_rng_state(void) {
    return rng_state;
}
API const uint32_t *native_memory(void) { return vmem_words; }
API const uint8_t *native_input_prev(void) { return input_prev; }
API uint32_t native_work_count(void) { return work_count; }
API uint32_t native_work_at(uint32_t i) { return i < work_count ? work[i] : UINT32_MAX; }
API uint64_t native_tick_count(void) { return tick_count; }
API uint32_t native_clock_value(void) { return clock_value; }
API uint32_t native_clock_until(void) { return clock_ticks_until_toggle; }
API uint32_t native_vmem_address(void) { return vmem_address_value; }
API uint32_t native_vmem_loaded(void) { return vmem_loaded_address; }
API uint32_t native_vmem_data(void) { return vmem_data_value; }
API uint32_t native_vmem_lock(void) { return vmem_lock_pending; }
API uint32_t native_vmem_deferred(void) { return vmem_deferred_publish; }
