from app.ontology.mapper import OntologyMapper


def test_ontology_mapper_fallback():
    mapper = OntologyMapper()
    mappings = mapper.map_phrase("diabetes")
    assert isinstance(mappings, list)
    assert any(m.get("ontology") == "SNOMEDCT" for m in mappings)


def test_ontology_mapper_cache():
    mapper = OntologyMapper()
    first = mapper.map_phrase("metformin")
    second = mapper.map_phrase("metformin")
    assert first == second
