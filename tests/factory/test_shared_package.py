"""The shared ``llmforge_common`` package and the shims that point at it.

The microservices used to carry byte-identical copies of the same modules
(``gpu_client.py``, ``service_orchestrator.py``, ``logging.py`` ...). There is now
one copy, in ``src/shared/llmforge_common``, and each old file is a shim of a few
lines that makes the old import path resolve to the package module. Three things
keep that true:

(a) no two Python files of 20 lines or more under ``src/`` or ``models/`` are
    byte-identical, so a copy cannot quietly come back;
(b) every shim names a module that exists in the package;
(c) loaded the way a service loads it (the service's ``src`` directory first on
    ``sys.path``), each shim *is* the package module and has the names the service
    imports from it.

(c) imports real service modules, so it runs in a subprocess per service: nothing
leaks into this test session, and a module whose third-party imports are not
installed here is skipped with the name of what is missing rather than failed.
On a machine without aiohttp that skips every GPU and orchestrator module; in an
image built from src/docker/shared-base they load.
"""

import ast
import functools
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src" / "shared" / "llmforge_common"
PACKAGE_DIR = PACKAGE_ROOT / "llmforge_common"
SERVICES_DIR = ROOT / "src" / "microservices"
SCAN_DIRS = ("src", "models")

# A shim is 13 to 17 lines, and the shims for one module are identical to each
# other by design. Twenty lines is the smallest file that counts as a real copy.
MIN_LINES = 20

SHIM_IMPORT = re.compile(r"^\s*from llmforge_common import (\w+) as _impl\s*$", re.MULTILINE)

# Directories that hold build output, not source, when git cannot tell us.
_NOT_SOURCE = {"__pycache__", "build", "dist", "node_modules", ".git"}


# ---- which files are looked at ---------------------------------------------

@functools.lru_cache(maxsize=None)
def _python_files():
    """Every Python source file under src/ and models/: tracked, or new and not ignored.

    New files are included so a copy is caught before it is committed, not after.
    Without git (a source archive, a container) the directories are walked instead.
    """
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *SCAN_DIRS],
            cwd=ROOT, check=True, capture_output=True, text=True, timeout=60).stdout.split("\0")
        files = [ROOT / name for name in listed if name.endswith(".py")]
    except (OSError, subprocess.SubprocessError):
        files = []
        for top in SCAN_DIRS:
            for directory, subdirs, names in os.walk(ROOT / top):
                subdirs[:] = [d for d in subdirs if d not in _NOT_SOURCE and not d.endswith(".egg-info")]
                files += [Path(directory) / name for name in names if name.endswith(".py")]
    # A file deleted from the working tree is still in the index; skip it.
    return tuple(sorted(path for path in files if path.is_file()))


def _identical_groups(paths, min_lines=MIN_LINES):
    """Groups of two or more byte-identical files with at least ``min_lines`` lines."""
    by_digest = {}
    for path in paths:
        data = path.read_bytes()
        if len(data.splitlines()) >= min_lines:
            by_digest.setdefault(hashlib.sha256(data).hexdigest(), []).append(path)
    return sorted((group for group in by_digest.values() if len(group) > 1), key=lambda g: str(g[0]))


def _relative(path):
    return path.relative_to(ROOT).as_posix()


# ---- (a) no byte-identical copies ------------------------------------------

def test_duplicate_detector_counts_only_files_of_twenty_lines_or_more(tmp_path):
    long_text = "".join(f"value_{i} = {i}\n" for i in range(MIN_LINES))
    short_text = "".join(f"value_{i} = {i}\n" for i in range(MIN_LINES - 1))
    for name, text in (("a.py", long_text), ("b.py", long_text), ("c.py", long_text + "extra = 1\n"),
                       ("short_a.py", short_text), ("short_b.py", short_text)):
        (tmp_path / name).write_text(text)
    groups = _identical_groups(sorted(tmp_path.iterdir()))
    assert [[p.name for p in group] for group in groups] == [["a.py", "b.py"]]


def test_no_two_python_files_under_src_or_models_are_byte_identical():
    files = _python_files()
    # The services alone hold more than this; a listing that comes back nearly
    # empty means the scan is broken, and an empty scan would pass for no reason.
    assert len(files) > 80, f"only {len(files)} Python files found under {SCAN_DIRS}"
    groups = _identical_groups(files)
    report = "\n".join("  identical: " + ", ".join(_relative(p) for p in group) for group in groups)
    assert not groups, (
        "byte-identical Python files (keep one copy in src/shared/llmforge_common and "
        "turn the others into shims):\n" + report)


# ---- (b) every shim points at a module that exists -------------------------

@functools.lru_cache(maxsize=None)
def _shims():
    """(path, package module name) for every shim under src/ and models/."""
    found = []
    for path in _python_files():
        if PACKAGE_ROOT in path.parents:
            continue
        match = SHIM_IMPORT.search(path.read_text(encoding="utf-8"))
        if match:
            found.append((path, match.group(1)))
    return tuple(found)


def _package_modules():
    return sorted(p.stem for p in PACKAGE_DIR.glob("*.py") if p.stem != "__init__")


def test_shims_were_found_and_every_package_module_is_used_by_a_service():
    shims = _shims()
    assert shims, "no shims found: has the shim import line changed shape?"
    used_by_services = {module for path, module in shims if SERVICES_DIR in path.parents}
    # A module no service reaches through a shim is dead weight in every image.
    assert used_by_services == set(_package_modules())


@pytest.mark.parametrize("path,module", _shims(), ids=[_relative(p) for p, _ in _shims()])
def test_shim_names_a_module_that_exists_in_the_package(path, module):
    assert (PACKAGE_DIR / f"{module}.py").is_file(), f"{_relative(path)} points at llmforge_common.{module}"
    # The shim keeps the file name of the module it replaced, so the mapping from
    # an old import path to the package module can be read off without opening it.
    assert path.stem == module
    text = path.read_text(encoding="utf-8")
    assert "sys.modules[__name__] = _impl" in text
    assert len(text.splitlines()) < MIN_LINES, "a shim must stay smaller than a file the duplicate check counts"
    # If the package is missing the shim must say so, not fail with a bare "No module named".
    assert "llmforge_common is not installed" in text


def test_the_package_imports_nothing_at_package_level():
    # A light service that needs only utils.performance must not import, or fail
    # on, torch or aiohttp because another module in the package needs them.
    tree = ast.parse((PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8"))
    assert not [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]


# ---- (c) loaded as a service loads it --------------------------------------

def _import_roots():
    """Where each tree's absolute imports start: name -> directories put first on sys.path.

    A service runs as ``python src/main.py`` (or ``python main.py`` inside src), so
    its ``src`` directory is ``sys.path[0]``. The legacy top-level copies were
    written for a process started at the repository root with ``src`` importable.
    """
    roots = {service.name: (service / "src",)
             for service in sorted(SERVICES_DIR.iterdir()) if (service / "src").is_dir()}
    roots["legacy"] = (ROOT / "src", ROOT)
    return roots


def _dotted(path, roots):
    """The import name of ``path`` under the first root that contains it, or None."""
    for root in roots:
        if root in path.parents:
            parts = path.relative_to(root).with_suffix("").parts
            # src/microservices and src/shared are not import packages of the legacy tree.
            if root == ROOT / "src" and parts[0] in ("microservices", "shared", "docker"):
                return None
            if root == ROOT and parts[0] != "models":
                return None
            return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    return None


def _imported_names(source_file, importer_name, is_package):
    """{module: {names}} for every ``from module import names`` in one file, relative ones resolved."""
    wanted = {}
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            if importer_name is None:
                continue
            base = importer_name.split(".")
            base = base if is_package else base[:-1]
            base = base[:len(base) - (node.level - 1)]
            module = ".".join(base + ([node.module] if node.module else []))
        else:
            module = node.module
        wanted.setdefault(module, set()).update(alias.name for alias in node.names if alias.name != "*")
    return wanted


@functools.lru_cache(maxsize=None)
def _shim_plan():
    """{tree: {shim import name: (package module, sorted names the tree imports from it)}}."""
    plan = {}
    for tree_name, roots in _import_roots().items():
        shims = {}
        for path, module in _shims():
            name = _dotted(path, roots)
            if name is not None:
                shims[name] = module
        if not shims:
            continue
        # Names come from the tree's own files and from the package modules, which
        # import utils.logging and models.schemas through the same service paths.
        wanted = {}
        sources = [p for p in _python_files() if _dotted(p, roots) is not None]
        for source in sources:
            found = _imported_names(source, _dotted(source, roots), source.name == "__init__.py")
            for module, names in found.items():
                wanted.setdefault(module, set()).update(names)
        for source in PACKAGE_DIR.glob("*.py"):
            for module, names in _imported_names(source, None, False).items():
                wanted.setdefault(module, set()).update(names)
        plan[tree_name] = {name: (module, sorted(wanted.get(name, ()))) for name, module in sorted(shims.items())}
    return plan


# Runs inside the subprocess. Every shim of one tree is imported in one process,
# as the service itself does, and the outcome of each is reported separately.
_LOADER = r"""
import importlib, json, sys, traceback
roots, package_root, plan = json.loads(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
# The roots first, as the script directory is for a service, then this checkout's
# package ahead of any copy that happens to be installed.
sys.path[0:0] = roots + [package_root]
results = {}
for name, (target, names) in plan.items():
    try:
        module = importlib.import_module(name)
        impl = importlib.import_module("llmforge_common." + target)
        namespace = {}
        exec("import " + name, namespace)
        attribute = namespace[name.split(".")[0]]
        for part in name.split(".")[1:]:
            attribute = getattr(attribute, part)
        # "import a.b" leaves a.b as an attribute of a. That is the module, unless
        # a/__init__.py rebinds the name: the legacy config/__init__.py does
        # "from .settings import settings", so config.settings is the settings
        # object there, as it was before the move.
        leaf = name.split(".")[-1]
        attribute_ok = attribute is impl or attribute is getattr(impl, leaf, None)
        missing = []
        for wanted in names:
            try:
                exec("from %s import %s" % (name, wanted), namespace)
            except ImportError:
                missing.append(wanted)
                continue
            if namespace[wanted] is not getattr(impl, wanted):
                missing.append(wanted + " (a different object)")
        results[name] = {"is_package_module": module is impl and sys.modules[name] is impl and attribute_ok,
                         "module_name": module.__name__, "missing": missing}
    except ModuleNotFoundError as exc:
        results[name] = {"missing_module": exc.name, "error": traceback.format_exc()}
    except Exception:
        results[name] = {"error": traceback.format_exc()}
print("\n@@RESULT@@" + json.dumps(results))
"""


@functools.lru_cache(maxsize=None)
def _load_tree(tree_name):
    roots = _import_roots()[tree_name]
    # Some of these modules act on import (research_logger creates logs/research,
    # model_config looks for config/model_pairs.yaml), all relative to the working
    # directory. An empty temporary directory keeps that out of the repository.
    with tempfile.TemporaryDirectory(prefix="llmforge_shim_") as workdir:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run(
            [sys.executable, "-c", _LOADER, json.dumps([str(r) for r in roots]), str(PACKAGE_ROOT),
             json.dumps(_shim_plan()[tree_name])],
            cwd=workdir, env=env, capture_output=True, text=True, timeout=600)
    marker = done.stdout.rfind("@@RESULT@@")
    assert done.returncode == 0 and marker >= 0, (
        f"loading the {tree_name} shims crashed:\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}")
    return json.loads(done.stdout[marker + len("@@RESULT@@"):])


def _is_first_party(module_name, roots):
    top = module_name.split(".")[0]
    return top == "llmforge_common" or any((root / top).is_dir() or (root / f"{top}.py").is_file() for root in roots)


# The legacy top-level ``models`` package could never be imported: models/__init__.py
# does ``from .entities import *`` and there has never been a models/entities.py.
# That was so before the shims and is not theirs to fix. The first import of
# ``models`` fails that way; a later ``from models.schemas import ...`` in the same
# process can succeed, because the submodule is already in sys.modules by then. So
# for these two legacy names the old failure is accepted, and a load is checked.
_LEGACY_ALREADY_BROKEN = {"models.schemas": "models.entities", "utils.research_logger": "models.entities"}

_LOAD_CASES = [(tree, name) for tree, shims in _shim_plan().items() for name in shims]


@pytest.mark.parametrize("tree,name", _LOAD_CASES, ids=[f"{tree}:{name}" for tree, name in _LOAD_CASES])
def test_shim_loads_as_the_package_module_with_the_names_the_service_imports(tree, name):
    module, names = _shim_plan()[tree][name]
    result = _load_tree(tree)[name]
    roots = _import_roots()[tree]

    missing_module = result.get("missing_module")
    if missing_module:
        if tree == "legacy" and _LEGACY_ALREADY_BROKEN.get(name) == missing_module:
            pytest.skip(f"legacy {name}: models/__init__.py imports models/entities.py, which never existed; "
                        "it failed the same way before the move")
        top = missing_module.split(".")[0]
        if not _is_first_party(missing_module, roots) and importlib.util.find_spec(top) is None:
            # The import may be the module's own or one made by the service package
            # around it (services/__init__.py pulls in the whole ML stack in some).
            pytest.skip(f"loading {name} (llmforge_common.{module}) in {tree} needs {top}, "
                        "which is not installed here")
    assert "error" not in result, f"{tree}: importing {name} failed:\n{result.get('error')}"

    assert result["is_package_module"], f"{tree}: {name} did not resolve to llmforge_common.{module}"
    assert result["module_name"] == f"llmforge_common.{module}"
    assert not result["missing"], f"{tree} imports {result['missing']} from {name}; llmforge_common.{module} lacks them"


def test_each_service_tree_actually_had_its_imports_collected():
    # Guards the name collection: if the AST walk found nothing, (c) would only
    # check module identity and say nothing about the names services rely on.
    plan = _shim_plan()
    assert "get_logger" in plan["api_gateway"]["utils.logging"][1]
    assert "register_with_orchestrator" in plan["api_gateway"]["services.service_orchestrator"][1]
    assert {"FineTuneRequest", "TrainingStatus"} <= set(plan["training_service"]["models.schemas"][1])
    # research_logger lives in the package but imports the service's own schemas.
    assert "ResearchLogEntry" in plan["chat_comparison_service"]["models.schemas"][1]


def test_a_missing_package_is_reported_by_name():
    """Without llmforge_common the shim must say what to do, not just "No module named"."""
    shim = SERVICES_DIR / "api_gateway" / "src" / "utils" / "performance.py"
    code = ("import importlib.abc, sys\n"
            "class Hide(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'llmforge_common':\n"
            "            raise ModuleNotFoundError(\"No module named 'llmforge_common'\", name='llmforge_common')\n"
            "sys.meta_path.insert(0, Hide())\n"
            f"sys.path.insert(0, {str(shim.parents[1])!r})\n"
            "try:\n"
            "    import utils.performance\n"
            "except ImportError as exc:\n"
            "    print('MESSAGE:', exc)\n")
    with tempfile.TemporaryDirectory(prefix="llmforge_shim_") as workdir:
        done = subprocess.run([sys.executable, "-c", code], cwd=workdir, capture_output=True, text=True, timeout=120,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert "MESSAGE: llmforge_common is not installed" in done.stdout, done.stdout + done.stderr
    assert "pip install -e src/shared/llmforge_common" in done.stdout
