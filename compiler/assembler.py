from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

WORD_COUNT = 1 << 20
WORD_MASK = 0xFFFFFFFF
DEFAULT_ORIGIN = 0x0001
IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")
MACRO_RE = re.compile(r"^macro\s+([A-Za-z_]\w*)\s*\{(.*)\}\s*$")
PLACEHOLDER_RE = re.compile(r"\b([A-Za-z_]\w*)\?")


class AssemblyCompileError(ValueError):
    """Raised when VCB assembly cannot be parsed or assembled."""


@dataclass(frozen=True)
class MacroDefinition:
    name: str
    params: Tuple[str, ...]
    expression: str
    line_no: int


@dataclass(frozen=True)
class PendingWord:
    address: int
    expression: str
    line_no: int
    statement: str


def project_uses_assembly(project_data: dict) -> bool:
    if project_data.get("assembly_is_external"):
        assembly_ref = project_data.get("assembly")
        if not isinstance(assembly_ref, str) or not assembly_ref.strip():
            raise ValueError("External assembly projects must store the assembly file path in 'assembly'.")
        return True

    assembly = project_data.get("assembly")
    if assembly is None:
        return False
    if not isinstance(assembly, str):
        raise ValueError("Project 'assembly' must be stored as a string.")
    return assembly.strip() != ""


def load_project_assembly_source(project_data: dict, project_path: str | Path) -> tuple[str, str]:
    assembly = project_data.get("assembly", "")
    if not isinstance(assembly, str):
        raise ValueError("Project 'assembly' must be stored as a string.")

    if not project_data.get("assembly_is_external"):
        return assembly, str(project_path)

    source_path = Path(assembly)
    if not source_path.is_absolute():
        source_path = Path(project_path).resolve().parent / source_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Assembly source file not found: {source_path}")
    return source_path.read_text(encoding="utf-8"), str(source_path)


def assemble_project_assembly_words(
    project_data: dict,
    project_path: str | Path,
    *,
    word_count: int = WORD_COUNT,
) -> Dict[int, int]:
    source, source_name = load_project_assembly_source(project_data, project_path)
    return assemble_source_words(source, source_name=source_name, word_count=word_count)


def assemble_source_words(
    source: str,
    *,
    source_name: str = "<assembly>",
    word_count: int = WORD_COUNT,
) -> Dict[int, int]:
    assembler = _AssemblyAssembler(source=source, source_name=source_name, word_count=word_count)
    return assembler.assemble_words()


class _AssemblyAssembler:
    def __init__(self, *, source: str, source_name: str, word_count: int):
        self.source = source
        self.source_name = source_name
        self.word_count = word_count
        self.current_address = DEFAULT_ORIGIN
        self.symbols: Dict[str, int] = {}
        self.macros: Dict[str, MacroDefinition] = {}
        self.pending_words: List[PendingWord] = []

    def assemble_words(self) -> Dict[int, int]:
        for line_no, statement in self._iter_statements():
            self._process_statement(line_no, statement)

        word_map: Dict[int, int] = {}
        for pending in self.pending_words:
            value = self._evaluate_statement(pending.expression, pending.line_no, pending.statement)
            word_map[pending.address] = value & WORD_MASK
        return word_map

    def _iter_statements(self) -> Iterable[tuple[int, str]]:
        for line_no, raw_line in enumerate(self.source.splitlines(), start=1):
            stripped_comment = self._strip_comment(raw_line)
            if not stripped_comment.strip():
                continue
            for statement in self._split_statements(stripped_comment, line_no):
                if statement:
                    yield line_no, statement

    def _process_statement(self, line_no: int, statement: str) -> None:
        if statement.startswith("@"):
            label = statement[1:].strip()
            self._validate_identifier(label, line_no, "label")
            self._define_name(label, self.current_address, line_no, kind="label")
            return

        if statement.startswith("bookmark"):
            return

        macro_match = MACRO_RE.match(statement)
        if macro_match:
            macro_name = macro_match.group(1)
            macro_body = macro_match.group(2).strip()
            self._validate_identifier(macro_name, line_no, "macro")
            self._ensure_name_available(macro_name, line_no, "macro")
            params = self._extract_macro_params(macro_body)
            transformed_body = self._transform_macro_expression(macro_body)
            self.macros[macro_name] = MacroDefinition(
                name=macro_name,
                params=params,
                expression=transformed_body,
                line_no=line_no,
            )
            return

        parts = statement.split(None, 2)
        if parts[:1] == ["symbol"]:
            if len(parts) != 3:
                self._fail(line_no, "Expected 'symbol <name> <expression>'.", statement)
            name, expression = parts[1], parts[2]
            self._validate_identifier(name, line_no, "symbol")
            self._ensure_name_available(name, line_no, "symbol")
            value = self._evaluate_statement(expression, line_no, statement)
            self.symbols[name] = value & WORD_MASK
            return

        if parts[:1] == ["origin"]:
            if len(parts) != 2:
                self._fail(line_no, "Expected 'origin <address>'.", statement)
            address = self._evaluate_statement(parts[1], line_no, statement)
            if address >= self.word_count:
                self._fail(
                    line_no,
                    f"Origin address 0x{address:08X} is outside the valid VMEM range 0..0x{self.word_count - 1:05X}.",
                    statement,
                )
            self.current_address = address
            return

        pointer_parts = statement.split(None, 3)
        if pointer_parts[:1] == ["pointer"]:
            if len(pointer_parts) != 4 or pointer_parts[2] != "inline":
                self._fail(line_no, "Expected 'pointer <name> inline <expression>'.", statement)
            name = pointer_parts[1]
            self._validate_identifier(name, line_no, "pointer")
            self._ensure_name_available(name, line_no, "pointer")
            self._reserve_current_address(line_no, statement)
            self.symbols[name] = self.current_address
            self.pending_words.append(
                PendingWord(
                    address=self.current_address,
                    expression=pointer_parts[3],
                    line_no=line_no,
                    statement=statement,
                )
            )
            self.current_address += 1
            return

        self._reserve_current_address(line_no, statement)
        self.pending_words.append(
            PendingWord(
                address=self.current_address,
                expression=statement,
                line_no=line_no,
                statement=statement,
            )
        )
        self.current_address += 1

    def _reserve_current_address(self, line_no: int, statement: str) -> None:
        if self.current_address >= self.word_count:
            self._fail(
                line_no,
                f"Address 0x{self.current_address:08X} is outside the valid VMEM range 0..0x{self.word_count - 1:05X}.",
                statement,
            )

    def _define_name(self, name: str, value: int, line_no: int, *, kind: str) -> None:
        self._ensure_name_available(name, line_no, kind)
        self.symbols[name] = value & WORD_MASK

    def _ensure_name_available(self, name: str, line_no: int, kind: str) -> None:
        if name in self.symbols or name in self.macros:
            self._fail(line_no, f"Duplicate {kind} name '{name}'.", name)

    def _validate_identifier(self, name: str, line_no: int, kind: str) -> None:
        if not IDENTIFIER_RE.fullmatch(name):
            self._fail(line_no, f"Invalid {kind} name '{name}'.", name)

    def _evaluate_statement(self, expression: str, line_no: int, statement: str) -> int:
        head, *tail = expression.split()
        macro = self.macros.get(head)
        if macro is not None:
            if len(tail) != len(macro.params):
                self._fail(
                    line_no,
                    f"Macro '{macro.name}' expects {len(macro.params)} argument(s), got {len(tail)}.",
                    statement,
                )
            context = dict(self.symbols)
            for param_name, arg_expression in zip(macro.params, tail):
                placeholder_name = self._placeholder_name(param_name)
                context[placeholder_name] = self._evaluate_expression(arg_expression, self.symbols, line_no, statement)
            return self._evaluate_expression(macro.expression, context, line_no, statement)
        return self._evaluate_expression(expression, self.symbols, line_no, statement)

    def _evaluate_expression(self, expression: str, names: Dict[str, int], line_no: int, statement: str) -> int:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            self._fail(line_no, f"Invalid expression '{expression}': {exc.msg}.", statement)
        evaluator = _ExpressionEvaluator(
            names=names,
            line_no=line_no,
            statement=statement,
            source_name=self.source_name,
        )
        return evaluator.visit(tree)

    def _extract_macro_params(self, expression: str) -> Tuple[str, ...]:
        ordered: List[str] = []
        seen = set()
        for match in PLACEHOLDER_RE.finditer(expression):
            param_name = match.group(1)
            if param_name in seen:
                continue
            ordered.append(param_name)
            seen.add(param_name)
        return tuple(ordered)

    def _transform_macro_expression(self, expression: str) -> str:
        return PLACEHOLDER_RE.sub(lambda match: self._placeholder_name(match.group(1)), expression)

    @staticmethod
    def _placeholder_name(name: str) -> str:
        return f"__macro_param_{name}"

    def _strip_comment(self, raw_line: str) -> str:
        comment_index = raw_line.find("#")
        if comment_index == -1:
            return raw_line
        return raw_line[:comment_index]

    def _split_statements(self, line: str, line_no: int) -> List[str]:
        statements: List[str] = []
        current: List[str] = []
        brace_depth = 0
        for char in line:
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth < 0:
                    self._fail(line_no, "Unmatched closing brace in macro definition.", line.strip())
            if char == ";" and brace_depth == 0:
                statement = "".join(current).strip()
                if statement:
                    statements.append(statement)
                current = []
                continue
            current.append(char)

        if brace_depth != 0:
            self._fail(line_no, "Unclosed brace in macro definition.", line.strip())

        final_statement = "".join(current).strip()
        if final_statement:
            statements.append(final_statement)
        return statements

    def _fail(self, line_no: int, message: str, statement: str) -> None:
        raise AssemblyCompileError(f"{self.source_name}:{line_no}: {message} Statement: {statement}")


class _ExpressionEvaluator(ast.NodeVisitor):
    def __init__(self, *, names: Dict[str, int], line_no: int, statement: str, source_name: str):
        self.names = names
        self.line_no = line_no
        self.statement = statement
        self.source_name = source_name

    def visit_Expression(self, node: ast.Expression) -> int:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> int:
        if node.id not in self.names:
            self._fail(f"Unknown symbol '{node.id}'.")
        return self.names[node.id] & WORD_MASK

    def visit_Constant(self, node: ast.Constant) -> int:
        if not isinstance(node.value, int):
            self._fail(f"Unsupported constant {node.value!r}.")
        return node.value & WORD_MASK

    def visit_Num(self, node: ast.Num) -> int:
        return int(node.n) & WORD_MASK

    def visit_UnaryOp(self, node: ast.UnaryOp) -> int:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand & WORD_MASK
        if isinstance(node.op, ast.USub):
            return (-operand) & WORD_MASK
        if isinstance(node.op, ast.Invert):
            return (~operand) & WORD_MASK
        self._fail(f"Unsupported unary operator '{ast.dump(node.op)}'.")

    def visit_BinOp(self, node: ast.BinOp) -> int:
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return (left + right) & WORD_MASK
        if isinstance(node.op, ast.Sub):
            return (left - right) & WORD_MASK
        if isinstance(node.op, ast.Mult):
            return (left * right) & WORD_MASK
        if isinstance(node.op, ast.FloorDiv) or isinstance(node.op, ast.Div):
            if right == 0:
                self._fail("Division by zero.")
            return (left // right) & WORD_MASK
        if isinstance(node.op, ast.Mod):
            if right == 0:
                self._fail("Modulo by zero.")
            return (left % right) & WORD_MASK
        if isinstance(node.op, ast.LShift):
            return (left << right) & WORD_MASK
        if isinstance(node.op, ast.RShift):
            return (left >> right) & WORD_MASK
        if isinstance(node.op, ast.BitOr):
            return (left | right) & WORD_MASK
        if isinstance(node.op, ast.BitAnd):
            return (left & right) & WORD_MASK
        if isinstance(node.op, ast.BitXor):
            return (left ^ right) & WORD_MASK
        self._fail(f"Unsupported binary operator '{ast.dump(node.op)}'.")

    def generic_visit(self, node: ast.AST) -> int:
        self._fail(f"Unsupported expression node '{ast.dump(node)}'.")

    def _fail(self, message: str) -> None:
        raise AssemblyCompileError(f"{self.source_name}:{self.line_no}: {message} Statement: {self.statement}")
