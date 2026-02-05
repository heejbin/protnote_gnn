"""Generate parenthood JSON for GO and EC label hierarchies.

Replaces the proteinfer parenthood_bin.py + parenthood_lib.py dependency.
Uses obonet (already a project dependency) for GO and ports the EC hierarchy
logic with pure Python (no TensorFlow).

Output: JSON mapping each GO/EC term to its sorted list of transitive parent
labels (including itself for canonical terms, excluding itself for alt_ids
and obsolete terms with replacements).
"""

import argparse
import json
import logging
import re
from pathlib import Path

import networkx as nx
import obonet
from tqdm import tqdm

from protnote.utils.configs import get_project_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── GO parenthood (via obonet + networkx) ────────────────────────────────

def build_go_parenthood(obo_path: str) -> dict:
    """Parse a GO .obo file and compute transitive parenthood for all terms.

    Matches the behavior of proteinfer's parenthood_lib:
    - is_a, replaced_by, and alt_id relations are all followed transitively
    - Canonical terms include themselves in their parent set
    - alt_id and replaced_by terms do NOT include themselves
    """
    logger.info(f"Reading GO ontology from {obo_path}")
    full_graph = obonet.read_obo(obo_path, ignore_obsolete=False)
    logger.info(
        f"Loaded {full_graph.number_of_nodes()} GO terms, "
        f"{full_graph.number_of_edges()} total edges"
    )

    # obonet includes part_of, regulates, etc. as edges — proteinfer only
    # follows is_a for parenthood. Build a clean DiGraph with only is_a edges.
    graph = nx.DiGraph()
    graph.add_nodes_from(full_graph.nodes(data=True))
    for u, v, key in full_graph.edges(keys=True):
        if key == "is_a":
            graph.add_edge(u, v)
    logger.info(f"Filtered to {graph.number_of_edges()} is_a edges")

    # Track non-canonical terms (should not include self in parent set)
    non_canonical = set()

    # Add replaced_by as edges so transitive closure follows them
    replaced_count = 0
    for node, data in graph.nodes(data=True):
        for replacement in data.get("replaced_by", []):
            if replacement in graph:
                graph.add_edge(node, replacement)
                non_canonical.add(node)
                replaced_count += 1

    # Add alt_id virtual nodes with edges to their canonical terms
    alt_count = 0
    alt_edges = []
    for node, data in full_graph.nodes(data=True):
        for alt_id in data.get("alt_id", []):
            alt_edges.append((alt_id, node))
            non_canonical.add(alt_id)
            alt_count += 1

    for alt_id, canonical in alt_edges:
        if alt_id not in graph:
            graph.add_node(alt_id)
        graph.add_edge(alt_id, canonical)

    logger.info(f"Added {replaced_count} replaced_by edges, {alt_count} alt_id nodes")

    # Compute transitive parenthood for all nodes
    # Edges go child → parent, so nx.descendants() follows to ancestors
    parenthood = {}
    for node in tqdm(graph.nodes(), total=graph.number_of_nodes(), desc="GO parenthood"):
        try:
            parents = nx.descendants(graph, node)
        except nx.NetworkXError:
            parents = set()

        if node not in non_canonical:
            parents.add(node)

        parenthood[node] = parents

    canonical_count = sum(1 for n in parenthood if n not in non_canonical)
    logger.info(
        f"GO parenthood: {len(parenthood)} total terms "
        f"({canonical_count} canonical, {len(non_canonical)} non-canonical)"
    )
    return parenthood


# ─── EC parenthood (ported from proteinfer parenthood_lib) ────────────────

EC_NUMBER_REGEX = r"(\d+).([\d\-n]+).([\d\-n]+).([\d\-n]+)"
_TOP_LEVEL_EC_CLASS_VALUE = "-.-.-.-"
_NON_LEAF_NODE_LINE_REGEX = re.compile(r"^\d\.")


def _replace_one_level_up(s: str) -> str:
    """Find direct parent of an EC label by replacing the lowest specific level with '-'."""
    n_dashes = s.count("-")
    replacements = {
        0: r"\1.\2.\3.-",
        1: r"\1.\2.-.-",
        2: r"\1.-.-.-",
        3: "-.-.-.-",
    }
    if n_dashes not in replacements:
        raise ValueError(f"Unexpected number of dashes in EC number: {s}")
    return re.sub(EC_NUMBER_REGEX, replacements[n_dashes], s)


def _all_ec_parents(label: str) -> set:
    """Compute all parents of an EC label (including itself, excluding root -.-.-.-).

    E.g. '1.2.3.4' → {'1.2.3.4', '1.2.3.-', '1.2.-.-', '1.-.-.-'}
    """
    parent = label
    parents = set()
    while parent != _TOP_LEVEL_EC_CLASS_VALUE:
        parents.add(parent)
        parent = _replace_one_level_up(parent)
    return parents


def _parse_ec_leaf_nodes(enzyme_dat_contents: str) -> list:
    """Parse enzyme.dat (ftp.expasy.org/databases/enzyme/enzyme.dat) into (id, desc) pairs."""
    entries = []
    for block in enzyme_dat_contents.split("\nID")[1:]:
        lines = block.split("\n")
        term_id = re.findall(r"\s+(.*)", lines[0])[0]
        desc = ""
        for line in lines:
            if line.startswith("DE"):
                desc += re.findall(r"DE\s+(.*)", line)[0]
        entries.append((term_id, desc))
    return entries


def _parse_ec_non_leaf_nodes(enzclass_txt_contents: str) -> list:
    """Parse enzclass.txt (ftp.expasy.org/databases/enzyme/enzclass.txt) into (id, desc) pairs."""
    entries = []
    for line in enzclass_txt_contents.split("\n"):
        if _NON_LEAF_NODE_LINE_REGEX.match(line):
            line = line.strip()
            term_id = "".join(line[0:9]).replace(" ", "")
            term_desc = re.findall(r".*.-\s+(.*)", line)[0]
            entries.append((term_id, term_desc))
    return entries


def build_ec_parenthood(enzyme_dat_path: str, enzclass_txt_path: str) -> dict:
    """Build transitive EC parenthood dict from enzyme.dat and enzclass.txt.

    Returns dict mapping 'EC:X.Y.Z.W' to set of parent labels (including self).
    Also includes all intermediate non-leaf nodes as keys.
    """
    logger.info(f"Reading EC data from {enzyme_dat_path} and {enzclass_txt_path}")

    with open(enzyme_dat_path) as f:
        leaf_nodes = _parse_ec_leaf_nodes(f.read())
    with open(enzclass_txt_path) as f:
        non_leaf_nodes = _parse_ec_non_leaf_nodes(f.read())

    logger.info(f"Parsed {len(leaf_nodes)} leaf + {len(non_leaf_nodes)} non-leaf EC nodes")

    parenthood = {}
    for label, _ in tqdm(leaf_nodes + non_leaf_nodes, desc="EC parenthood"):
        parents = _all_ec_parents(label)
        # Also ensure all intermediate parents are in the dict
        for parent in parents:
            if "EC:" + parent not in parenthood:
                parent_parents = _all_ec_parents(parent)
                parenthood["EC:" + parent] = {"EC:" + p for p in parent_parents}

    logger.info(f"EC parenthood: {len(parenthood)} total terms")
    return parenthood


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    project_root = get_project_root()
    annotations_dir = project_root / "data" / "annotations"
    output_dir = project_root / "data" / "vocabularies"

    parser = argparse.ArgumentParser(
        description="Generate parenthood JSON for GO and/or EC label hierarchies."
    )
    parser.add_argument(
        "--go-obo",
        type=str,
        default=None,
        help="Path to GO .obo file. If not provided, uses latest .obo in data/annotations/.",
    )
    parser.add_argument(
        "--enzyme-dat",
        type=str,
        default=None,
        help="Path to enzyme.dat file. If not provided, looks in data/annotations/.",
    )
    parser.add_argument(
        "--enzclass-txt",
        type=str,
        default=None,
        help="Path to enzclass.txt file. If not provided, looks in data/annotations/.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Default: data/vocabularies/parenthood_{obo_date}.json",
    )
    parser.add_argument(
        "--skip-ec",
        action="store_true",
        help="Skip EC parenthood (only generate GO).",
    )
    args = parser.parse_args()

    # Resolve GO .obo path
    if args.go_obo:
        go_obo_path = Path(args.go_obo)
    else:
        obo_files = sorted(annotations_dir.glob("go_*.obo"))
        if not obo_files:
            parser.error(
                f"No .obo files found in {annotations_dir}. "
                "Provide --go-obo or run: python bin/download_GO_annotations.py"
            )
        go_obo_path = obo_files[-1]  # Latest by name
        logger.info(f"Auto-detected GO .obo file: {go_obo_path.name}")

    # Build GO parenthood
    parenthood = build_go_parenthood(str(go_obo_path))

    # Build EC parenthood (optional)
    if not args.skip_ec:
        enzyme_dat = Path(args.enzyme_dat) if args.enzyme_dat else None
        enzclass_txt = Path(args.enzclass_txt) if args.enzclass_txt else None

        # Auto-detect EC files
        if enzyme_dat is None:
            candidates = sorted(annotations_dir.glob("enzyme*.dat"))
            if candidates:
                enzyme_dat = candidates[-1]
                logger.info(f"Auto-detected enzyme.dat: {enzyme_dat.name}")
        if enzclass_txt is None:
            candidates = sorted(annotations_dir.glob("enzclass*.txt"))
            if candidates:
                enzclass_txt = candidates[-1]
                logger.info(f"Auto-detected enzclass.txt: {enzclass_txt.name}")

        if enzyme_dat and enzclass_txt and enzyme_dat.exists() and enzclass_txt.exists():
            ec_parenthood = build_ec_parenthood(str(enzyme_dat), str(enzclass_txt))

            # Check for key overlap
            overlap = set(parenthood.keys()) & set(ec_parenthood.keys())
            if overlap:
                raise ValueError(f"GO and EC keys overlap: {overlap}")

            parenthood.update(ec_parenthood)
        else:
            logger.warning(
                "EC source files not found (enzyme.dat / enzclass.txt). "
                "Skipping EC parenthood. Provide --enzyme-dat and --enzclass-txt, "
                "or place files in data/annotations/."
            )

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Derive name from .obo filename: go_2025-10-10.obo → parenthood_2025_10.json
        obo_stem = go_obo_path.stem  # e.g. "go_2025-10-10"
        date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", obo_stem)
        if date_match:
            y, m, d = date_match.groups()
            suffix = f"{y}_{m}"
        else:
            suffix = obo_stem.replace("go_", "")
        output_path = output_dir / f"parenthood_{suffix}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize: convert sets to sorted lists
    serializable = {k: sorted(v) for k, v in parenthood.items()}

    logger.info(f"Writing {len(serializable)} terms to {output_path}")
    with open(output_path, "w") as f:
        json.dump(serializable, f, sort_keys=True)

    logger.info("Done.")


if __name__ == "__main__":
    """
    Example usage:

    # Auto-detect source files, output to data/vocabularies/parenthood_{date}.json
    python bin/generate_parenthood.py

    # Specify files explicitly
    python bin/generate_parenthood.py \
        --go-obo data/annotations/go_2025-10-10.obo \
        --enzyme-dat data/annotations/enzyme.dat \
        --enzclass-txt data/annotations/enzclass.txt \
        --output data/vocabularies/parenthood_2025_10.json

    # GO only (skip EC)
    python bin/generate_parenthood.py --skip-ec

    """
    main()
