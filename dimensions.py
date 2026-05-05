"""Ward feature taxonomy and Ward departure scoring.

The taxonomy is shared between the extraction judge prompt (in
``alienbench.extract``) and the programmatic scoring in
``alienbench.score``; the ablation modules also depend on
:data:`DIMENSION_IDS` and :func:`compute_ward_score`.
"""

from __future__ import annotations


WARD_DIMENSIONS = [
    {
        "id": "symmetry",
        "label": "Body Symmetry",
        "earth_default": "bilateral symmetry (left-right mirror image)",
        "departure_examples": "radial symmetry, rotational symmetry, asymmetric, fractal, no consistent symmetry",
        "boundary_note": "Having 6 limbs arranged bilaterally is still bilateral — score 0. True departure requires a fundamentally different symmetry type.",
    },
    {
        "id": "sensory_organs",
        "label": "Sensory Organs",
        "earth_default": "eyes, ears, or nose located on a distinct head",
        "departure_examples": "no eyes, distributed sensing across the body, echolocation as primary sense, sensing electromagnetic fields, no distinct head region",
        "boundary_note": "Having eyes but in unusual locations still counts as Earth-typical if they're on a head structure — score 0. Departure requires a fundamentally different sensory mechanism or absence of head-based sensing.",
    },
    {
        "id": "locomotion",
        "label": "Locomotion",
        "earth_default": "uses legs for movement (2 or 4 legs)",
        "departure_examples": "no limbs, moves by rolling, flows like liquid, purely aerial, sessile (doesn't move), moves by jet propulsion",
        "boundary_note": "6 or 8 legs still uses legs — score 0. Departure means a fundamentally different locomotion mechanism, not just a different leg count.",
    },
    {
        "id": "body_plan",
        "label": "Body Plan",
        "earth_default": "distinct head, torso, and limbs",
        "departure_examples": "no distinct head, modular/segmented without central body, amorphous, colonial organism, entirely spherical or geometric",
        "boundary_note": "Unusual proportions still count as Earth-typical if the basic head/torso/limb structure is present — score 0.",
    },
    {
        "id": "skin_covering",
        "label": "Skin / Body Covering",
        "earth_default": "skin, fur, scales, or feathers",
        "departure_examples": "crystalline exoskeleton, metallic surface, gaseous body, energy-based form, transparent membrane, bioluminescent gel",
        "boundary_note": "Unusual skin colors or textures are still Earth-typical — score 0. Departure requires a fundamentally non-biological covering material.",
    },
    {
        "id": "reproduction",
        "label": "Reproduction",
        "earth_default": "sexual reproduction with binary sexes",
        "departure_examples": "asexual budding, fragmentation, spore release, more than two sexes, fusion of individuals, cloning",
        "boundary_note": "If reproduction is not described, score 0 (assume Earth default). Only score 1 if a non-binary-sexual mechanism is explicitly described.",
    },
    {
        "id": "metabolism",
        "label": "Metabolism / Energy Source",
        "earth_default": "eats organic matter and/or breathes oxygen",
        "departure_examples": "chemosynthesis, feeds on radiation or magnetism, absorbs minerals directly, photosynthesis as sole energy source, non-chemical energy processing",
        "boundary_note": "Eating unusual organic matter (e.g., rocks, other aliens) is still heterotrophy — score 0 only if the energy source is fundamentally non-organic-based.",
    },
    {
        "id": "communication",
        "label": "Communication",
        "earth_default": "sound-based or visual signals (vocalisation, body language)",
        "departure_examples": "chemical signalling only, electromagnetic pulses, direct neural link, vibration through substrate, no communication",
        "boundary_note": "If communication is not described, score 0. Score 1 only if a non-acoustic, non-visual mechanism is explicitly primary.",
    },
    {
        "id": "habitat",
        "label": "Habitat",
        "earth_default": "inhabits a solid planetary surface (land) or an open water body (ocean, lake, river)",
        "departure_examples": "permanently deep-subsurface (kilometres underground with no surface contact), permanently airborne or atmospheric with no surface or water contact, vacuum or space-dwelling, endoparasitic lifestyle entirely inside another organism, field-bound or energy-pattern entity with no material habitat",
        "boundary_note": "Surface-variant habitats remain Earth-typical and score 0, including: caves and shallow subsurface burrows on a planetary surface; deep-sea benthic zones; wetlands, marshes, and swamps; arid deserts; ice surfaces or subglacial liquid water; underwater environments with exotic chemistry (ammonia, methane, liquid CO2) which still count as water-like. Score 1 only when the habitat is fundamentally non-surface and non-aquatic: permanently deep-subsurface with no surface contact, permanently suspended in atmosphere without surface interaction, vacuum or open-space, strictly endoparasitic, or an energy-field entity with no material habitat.",
    },
    {
        "id": "cognition",
        "label": "Cognitive Architecture",
        "earth_default": "centralised brain or nervous system",
        "departure_examples": "distributed intelligence with no central brain, hive mind (intelligence only at colony level), non-neural processing, no apparent cognition",
        "boundary_note": "If cognition is not described, score 0. Score 1 only if a non-centralised or non-neural cognitive architecture is explicitly described.",
    },
]

DIMENSION_IDS = [d["id"] for d in WARD_DIMENSIONS]


def compute_ward_score(features: dict, dimension_ids: list[str] | None = None) -> dict:
    """Compute the Ward departure score from a parsed feature record.

    ``dimension_ids`` defaults to the full 10-dimension taxonomy. Passing a
    subset enables the Dimension Sensitivity ablation
    (paper sec:ablation_dimensions) to recompute totals over reduced sets.
    """
    dim_ids = dimension_ids if dimension_ids is not None else DIMENSION_IDS
    per_dimension = {
        dim_id: int(bool(features[dim_id]["is_departure"]))
        for dim_id in dim_ids
    }
    return {
        "per_dimension": per_dimension,
        "total": sum(per_dimension.values()),
    }
