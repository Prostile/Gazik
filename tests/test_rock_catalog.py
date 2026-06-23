from synthetic_well_logs.config.scenario import SUPPORTED_FACIES as CONFIG_SUPPORTED_FACIES
from synthetic_well_logs.rocks import PRIORS, SUPPORTED_FACIES, validate_rock_catalog


def test_supported_facies_derived_from_rock_catalog() -> None:
    assert frozenset(PRIORS) == SUPPORTED_FACIES
    assert CONFIG_SUPPORTED_FACIES == SUPPORTED_FACIES


def test_rock_catalog_is_complete() -> None:
    validate_rock_catalog()
