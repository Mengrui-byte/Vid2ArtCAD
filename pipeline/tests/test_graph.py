import unittest

from pipeline.kinematics.graph import canonical_joint_type, normalize_graph
from pipeline.fusion.adapter import source_joint_dof_spec


def joint(a, b, kind):
    return {
        "occurrence_one": a,
        "occurrence_two": b,
        "joint_motion": {"joint_type": kind},
    }


class GraphTest(unittest.TestCase):
 def test_fixed_components_and_loop_are_preserved(self):
    graph = normalize_graph(
        ["a", "b", "c", "d"],
        {
            "fixed": joint("a", "b", "RigidJointType"),
            "r1": joint("b", "c", "RevoluteJointType"),
            "r2": joint("c", "d", "SliderJointType"),
            "r3": joint("d", "a", "RevoluteJointType"),
        },
    )
    self.assertEqual(len(graph.link_ids), 3)
    self.assertEqual(len(graph.tree_joints), 2)
    self.assertEqual(graph.loop_joints, ("r3",))
    self.assertEqual(graph.component_of["a"], graph.component_of["b"])
    self.assertEqual(graph.dof_order, ("r1:0", "r2:0", "r3:0"))


 def test_disconnected_components_have_multiple_roots(self):
    graph = normalize_graph(
        ["a", "b", "c"],
        {"r": joint("a", "b", "RevoluteJointType")},
    )
    self.assertEqual(len(graph.roots), 2)


 def test_special_joint_types_have_explicit_dofs(self):
    self.assertEqual(canonical_joint_type("CylindricalJointType"), "cylindrical")
    graph = normalize_graph(["a", "b"], {"c": joint("a", "b", "CylindricalJointType")})
    self.assertEqual(graph.dof_order, ("c:0", "c:1"))


 def test_unknown_joint_type_fails_loudly(self):
    with self.assertRaisesRegex(ValueError, "unsupported source joint type"):
        normalize_graph(["a", "b"], {"x": joint("a", "b", "Unknown")})

 def test_fusion_joint_axes_are_expanded_by_type(self):
    raw = joint("a", "b", "PlanarJointType")
    raw["name"] = "Plane1"
    raw["geometry_or_origin_one"] = {
        "origin": {"x": 1, "y": 2, "z": 3},
        "primary_axis_vector": {"x": 0, "y": 0, "z": 1},
        "secondary_axis_vector": {"x": 1, "y": 0, "z": 0},
        "tertiary_axis_vector": {"x": 0, "y": 1, "z": 0},
    }
    spec = source_joint_dof_spec(raw)
    self.assertEqual(spec["type"], "planar")
    self.assertEqual(spec["dof_names"], ("Plane1:0", "Plane1:1", "Plane1:2"))
    self.assertEqual(spec["axes_in_joint"][0], [0.0, 0.0, 1.0])

 def test_fusion_geometry_falls_back_to_second_side(self):
    raw = joint("a", "b", "SliderJointType")
    raw["geometry_or_origin_two"] = {
        "origin": {"x": 0, "y": 0, "z": 0},
        "primary_axis_vector": {"x": 0, "y": 1, "z": 0},
    }
    spec = source_joint_dof_spec(raw)
    self.assertEqual(spec["coordinate_status"], "resolved")
    self.assertEqual(spec["axes_in_joint"], [[0.0, 1.0, 0.0]])

 def test_missing_fusion_geometry_is_reviewable(self):
    spec = source_joint_dof_spec(joint("a", "b", "RevoluteJointType"))
    self.assertEqual(spec["coordinate_status"], "unresolved")
    self.assertEqual(spec["axes_in_joint"], [])
