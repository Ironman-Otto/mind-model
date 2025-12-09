"""
seed_concepts.py (v2)

Adds a higher-level hub Concept "Animal" and wires IS_A edges from Dog and Cat.
"""
from __future__ import annotations
from typing import Dict
from mind_model.concepts.concept import Concept
from mind_model.concepts.feature_ensemble import FeatureEnsemble


def make_animal_concept() -> Concept:
    c = Concept("Animal", description="Superclass for animals")
    # Coarse features – morphology, mobility, biology (toy vectors for demo)
    c.add_ensemble(FeatureEnsemble("morphology", "vision", [1, 0, 0, 0]))
    c.add_ensemble(FeatureEnsemble("mobility", "vision", [0, 1, 0, 0]))
    c.add_ensemble(FeatureEnsemble("biology", "knowledge", [0, 0, 1, 0]))
    c.add_ensemble(FeatureEnsemble("word_animal", "language", [0, 0, 0, 1]))
    return c


def make_dog_concept() -> Concept:
    c = Concept("Dog", description="Canine animal concept")
    e_shape = FeatureEnsemble("shape_canine", "vision", [1, 0, 0, 0])
    e_color = FeatureEnsemble("color_brown", "vision", [0, 1, 0, 0])
    e_sound = FeatureEnsemble("sound_bark", "audio", [0, 0, 1, 0])
    e_label = FeatureEnsemble("word_dog", "language", [0, 0, 0, 1])
    e_shape.add_link(e_color.ensemble_id, 0.2)
    e_shape.add_link(e_sound.ensemble_id, 0.15)
    e_label.add_link(e_shape.ensemble_id, 0.25)
    for e in [e_shape, e_color, e_sound, e_label]:
        c.add_ensemble(e)
    return c


def make_cat_concept() -> Concept:
    c = Concept("Cat", description="Feline animal concept")
    e_shape = FeatureEnsemble("shape_feline", "vision", [1, 0, 0, 0])
    e_color = FeatureEnsemble("color_black", "vision", [0, 1, 0, 0])
    e_sound = FeatureEnsemble("sound_meow", "audio", [0, 0, 1, 0])
    e_label = FeatureEnsemble("word_cat", "language", [0, 0, 0, 1])
    e_shape.add_link(e_color.ensemble_id, 0.2)
    e_shape.add_link(e_sound.ensemble_id, 0.15)
    e_label.add_link(e_shape.ensemble_id, 0.25)
    for e in [e_shape, e_color, e_sound, e_label]:
        c.add_ensemble(e)
    return c


def make_car_concept() -> Concept:
    c = Concept("Car", description="Vehicle concept")
    e_shape = FeatureEnsemble("shape_vehicle", "vision", [1, 0, 0, 0])
    e_color = FeatureEnsemble("color_red", "vision", [0, 1, 0, 0])
    e_sound = FeatureEnsemble("sound_engine", "audio", [0, 0, 1, 0])
    e_label = FeatureEnsemble("word_car", "language", [0, 0, 0, 1])
    e_shape.add_link(e_color.ensemble_id, 0.2)
    e_shape.add_link(e_sound.ensemble_id, 0.15)
    e_label.add_link(e_shape.ensemble_id, 0.25)
    for e in [e_shape, e_color, e_sound, e_label]:
        c.add_ensemble(e)
    return c


def list_catalog() -> Dict[str, Concept]:
    """Return a catalog of ready concepts and wire IS_A relations to Animal."""
    animal = make_animal_concept()
    dog = make_dog_concept()
    cat = make_cat_concept()
    car = make_car_concept()

    # Wire IS_A edges (Cat → Animal, Dog → Animal)
    dog.add_relationship("IS_A", animal.concept_id, description="dog is an animal")
    cat.add_relationship("IS_A", animal.concept_id, description="cat is an animal")

    return {
        "Animal": animal,
        "Dog": dog,
        "Cat": cat,
        "Car": car,
    }
