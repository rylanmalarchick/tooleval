import Mathlib

/-!
# The global phase that tooleval's equivalence grader removes

`tooleval.qmlir.phase_distance(u, v)` computes the overlap `z = tr(u† v)` (numpy `vdot`),
sets `c = z / |z|` (or `c = 1` when `z = 0`), and returns `‖v - c • u‖`. The grader treats this
as the distance between `u` and `v` up to global phase. This file proves that the choice of `c`
is a minimizer: no unit phase gives a smaller distance. It holds in any complex inner product
space; with the Frobenius inner product, `inner ℂ u v = tr(u† v)`.
-/

open ComplexConjugate

/-- The phase the grader uses for the overlap `z`. -/
noncomputable def bestPhase (z : ℂ) : ℂ := if z = 0 then 1 else z / (‖z‖ : ℂ)

lemma norm_sub_smul_sq {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]
    (u v : E) (c : ℂ) (hc : ‖c‖ = 1) :
    ‖v - c • u‖ ^ 2 = ‖v‖ ^ 2 + ‖u‖ ^ 2 - 2 * (conj c * inner ℂ u v).re := by
  rw [norm_sub_sq (𝕜 := ℂ), inner_smul_right, norm_smul, hc, one_mul]
  have : (c * inner ℂ v u).re = (conj c * inner ℂ u v).re := by
    rw [← inner_conj_symm u v, ← Complex.conj_re (c * _)]
    simp [map_mul]
  rw [RCLike.re_to_complex, this]
  ring

theorem phase_distance_min {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]
    (u v : E) (c : ℂ) (hc : ‖c‖ = 1) :
    ‖v - bestPhase (inner ℂ u v) • u‖ ≤ ‖v - c • u‖ := by
  set z := inner ℂ u v with hz
  have hb : ‖bestPhase z‖ = 1 := by
    unfold bestPhase
    split_ifs with h
    · simp
    · rw [norm_div, Complex.norm_real, norm_norm, div_self (norm_ne_zero_iff.mpr h)]
  have key : (conj c * z).re ≤ (conj (bestPhase z) * z).re := by
    have h1 : (conj c * z).re ≤ ‖z‖ := by
      calc (conj c * z).re ≤ ‖conj c * z‖ := Complex.re_le_norm _
        _ = ‖z‖ := by rw [norm_mul, Complex.norm_conj, hc, one_mul]
    unfold bestPhase
    split_ifs with h
    · simp [h] at h1 ⊢
    · have : conj (z / (‖z‖ : ℂ)) * z = (‖z‖ : ℂ) := by
        have hn : (‖z‖ : ℂ) ≠ 0 := by exact_mod_cast norm_ne_zero_iff.mpr h
        rw [map_div₀, Complex.conj_ofReal, div_mul_eq_mul_div, mul_comm (conj z) z,
          Complex.mul_conj, Complex.normSq_eq_norm_sq]
        field_simp
        push_cast
        ring
      rw [this, Complex.ofReal_re]
      exact h1
  have hsq : ‖v - bestPhase z • u‖ ^ 2 ≤ ‖v - c • u‖ ^ 2 := by
    rw [norm_sub_smul_sq u v _ hb, norm_sub_smul_sq u v _ hc]
    linarith
  exact (pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).mp hsq
