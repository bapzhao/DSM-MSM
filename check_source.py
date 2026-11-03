#!/usr/bin/env python3
"""Check source provenance and small numerical identities; write no results."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dsm"))
import solver
import control
import benchmark
import run_burgers


class SourceChecks(unittest.TestCase):
    def test_extracted_functions_unchanged(self):
        manifest = json.loads((ROOT/"PROVENANCE.json").read_text())
        for item in manifest["files"]:
            if "functions" not in item:
                continue
            source = (ROOT/item["file"]).read_text()
            functions = {node.name: node for node in ast.parse(source).body
                         if isinstance(node, ast.FunctionDef)}
            for expected in item["functions"]:
                text = ast.get_source_segment(source, functions[expected["name"]])
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), expected["sha256"])

    def test_initial_normalization_and_reality(self):
        u = run_burgers.initial_state(512, 1536)
        self.assertAlmostEqual(control.kinetic_energy(u), 0.5, places=14)
        self.assertLess(control.hermitian_error(u), 1e-14)
        np.testing.assert_allclose(u, run_burgers.initial_state(512, 1536), atol=0, rtol=0)

    def test_fft_power_matches_explicit_folded_triads(self):
        n = 32
        k = solver.WaveNumber(2*np.pi, n)
        u = benchmark.random_hermitian_uk(n, 23)
        p, q, amplitude = control.phase_model(u, k)
        direct_p, direct_q = benchmark.direct_triad_pq_loop(u, k)
        np.testing.assert_allclose([p,q], [direct_p,direct_q], atol=3e-14, rtol=3e-14)
        for fraction in (0.0, -0.4, 0.8, 1.5):
            target = fraction*amplitude
            alpha, clipped, saturated = control.solve_alpha(p, q, target, n, None)
            actual = control.alias_power(control.translate(u,k,alpha),k)
            self.assertAlmostEqual(actual, clipped, places=13)
            self.assertEqual(saturated, abs(fraction)>1)

    def test_dealiased_convection_has_zero_total_power(self):
        cutoff = 16
        coarse = benchmark.random_hermitian_uk(2*cutoff, 42)
        u = run_burgers.project(coarse,cutoff,3*cutoff)
        k = solver.WaveNumber(2*np.pi,len(u))
        rhs = run_burgers.convection(u,k,cutoff)
        self.assertLess(abs(np.vdot(u,rhs).real), 2e-14)

    def test_original_split_update(self):
        cutoff = 16
        u = benchmark.random_hermitian_uk(2*cutoff, 42)
        k = solver.WaveNumber(2*np.pi,len(u))
        nu,dt = 0.005,2e-4
        expected = (u+dt*run_burgers.convection(u,k,cutoff))*(1-nu*k*k*dt)
        actual = solver.advance_step(u,k,nu,dt,np.abs(k)>=cutoff)
        np.testing.assert_allclose(actual,expected,atol=1e-16,rtol=1e-15)

    def test_msm_sampling_matrix(self):
        source = (ROOT/"msm/run_ns.py").read_text()
        tree = ast.parse(source)
        nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef)
                 and n.name in ("formula32_basis","formula32_epsilon","solve_alphaxy_from_formula32")]
        namespace = {"np":np}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),"MSM_functions","exec"),namespace)
        basis = namespace["formula32_basis"]
        pairs = [(0,0),(np.pi/2,0),(-np.pi/2,0),(0,np.pi/2),(0,-np.pi/2),
                 (np.pi/2,np.pi/2),(np.pi/2,-np.pi/2),(-np.pi/2,np.pi/2)]
        matrix = np.array([basis(x,y,1,1) for x,y in pairs])
        self.assertEqual(np.linalg.matrix_rank(matrix),8)
        self.assertAlmostEqual(np.linalg.det(matrix),-16,places=12)
        coefficients = np.array([0.8,-0.2,0.3,-0.1,0.2,0.5,0.1,-0.4])
        np.testing.assert_allclose(np.linalg.solve(matrix,matrix@coefficients),coefficients,
                                   atol=1e-14,rtol=1e-14)
        ax,ay,predicted,info = namespace["solve_alphaxy_from_formula32"](coefficients,32,32,0)
        self.assertEqual(ax,ay)
        self.assertLess(abs(predicted),1e-11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
