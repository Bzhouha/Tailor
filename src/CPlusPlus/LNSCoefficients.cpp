#include "LNSCoefficients.hpp"

#include <petscmath.h>

#include <algorithm>
#include <cmath>

namespace {

constexpr PetscReal oneThird = 1.0 / 3.0;
constexpr PetscReal twoThirds = 2.0 / 3.0;
constexpr PetscReal fourThirds = 4.0 / 3.0;

PetscReal component(const FlowState &state, FlowComponent field) {
  return state[static_cast<std::size_t>(flowIndex(field))];
}

bool finiteState(const FlowState &state) {
  return std::all_of(state.begin(), state.end(), [](PetscReal value) {
    return !PetscIsInfOrNanReal(value);
  });
}

} // namespace

PetscScalar &Block5::operator()(PetscInt row, PetscInt column) {
  return values_[static_cast<std::size_t>(row * flowComponentCount + column)];
}

const PetscScalar &Block5::operator()(PetscInt row, PetscInt column) const {
  return values_[static_cast<std::size_t>(row * flowComponentCount + column)];
}

bool Block5::finite() const noexcept {
  return std::all_of(values_.begin(), values_.end(), [](PetscScalar value) {
    return !PetscIsInfOrNanScalar(value);
  });
}

bool PhysicalLNSCoefficients::finite() const noexcept {
  return Gamma.finite() && A.finite() && B.finite() && C.finite() &&
         D.finite() && Vxx.finite() && Vxy.finite() && Vxz.finite() &&
         Vyy.finite() && Vyz.finite() && Vzz.finite();
}

const char *coefficientStatusName(CoefficientStatus status) noexcept {
  switch (status) {
  case CoefficientStatus::Success:
    return "success";
  case CoefficientStatus::NonFiniteInput:
    return "non-finite input";
  case CoefficientStatus::NonPositiveDensity:
    return "non-positive density";
  case CoefficientStatus::NonPositiveTemperature:
    return "non-positive temperature";
  case CoefficientStatus::NonFiniteOutput:
    return "non-finite output";
  }
  return "unknown";
}

CoefficientStatus makeBaseFlowPoint(const PetscScalar *state,
                                    const PetscScalar *derivatives,
                                    BaseFlowPoint &point) noexcept {
  point = {};
  for (PetscInt field = 0; field < flowComponentCount; ++field) {
    const PetscScalar stateValue = state[field];
    if (PetscIsInfOrNanScalar(stateValue) ||
        PetscImaginaryPart(stateValue) != 0.0)
      return CoefficientStatus::NonFiniteInput;
    point.value[static_cast<std::size_t>(field)] = PetscRealPart(stateValue);
    const auto assignDerivative = [&](BaseFlowDerivative derivative,
                                      FlowState &target) {
      const PetscScalar value = derivatives[baseFlowDerivativeIndex(
          derivative, static_cast<FlowComponent>(field))];
      if (PetscIsInfOrNanScalar(value) || PetscImaginaryPart(value) != 0.0)
        return false;
      target[static_cast<std::size_t>(field)] = PetscRealPart(value);
      return true;
    };
    if (!assignDerivative(BaseFlowDerivative::Dy, point.dy) ||
        !assignDerivative(BaseFlowDerivative::Dz, point.dz) ||
        !assignDerivative(BaseFlowDerivative::Dyy, point.dyy) ||
        !assignDerivative(BaseFlowDerivative::Dzz, point.dzz) ||
        !assignDerivative(BaseFlowDerivative::Dyz, point.dyz))
      return CoefficientStatus::NonFiniteInput;
  }
  if (!finiteState(point.value) || !finiteState(point.dy) ||
      !finiteState(point.dz) || !finiteState(point.dyy) ||
      !finiteState(point.dzz) || !finiteState(point.dyz))
    return CoefficientStatus::NonFiniteInput;
  if (component(point.value, FlowComponent::Density) <= 0.0)
    return CoefficientStatus::NonPositiveDensity;
  if (component(point.value, FlowComponent::Temperature) <= 0.0)
    return CoefficientStatus::NonPositiveTemperature;
  return CoefficientStatus::Success;
}

LNSCoefficients::LNSCoefficients(const Recipe &recipe)
    : reynolds_(recipe.reynolds), mach_(recipe.mach), prandtl_(recipe.prandtl),
      gamma_(recipe.ratioOfSpecificHeats),
      sutherlandRatio_(recipe.sutherlandConstant /
                       recipe.referenceTemperature) {}

CoefficientStatus LNSCoefficients::evaluate(
    const BaseFlowPoint &flow,
    PhysicalLNSCoefficients &coefficients) const noexcept {
  if (!finiteState(flow.value) || !finiteState(flow.dx) ||
      !finiteState(flow.dy) || !finiteState(flow.dz) ||
      !finiteState(flow.dxx) || !finiteState(flow.dyy) ||
      !finiteState(flow.dzz) || !finiteState(flow.dxy) ||
      !finiteState(flow.dxz) || !finiteState(flow.dyz))
    return CoefficientStatus::NonFiniteInput;

  const PetscReal rho = component(flow.value, FlowComponent::Density);
  const PetscReal U = component(flow.value, FlowComponent::U);
  const PetscReal V = component(flow.value, FlowComponent::V);
  const PetscReal W = component(flow.value, FlowComponent::W);
  const PetscReal T = component(flow.value, FlowComponent::Temperature);
  if (rho <= 0.0)
    return CoefficientStatus::NonPositiveDensity;
  if (T <= 0.0)
    return CoefficientStatus::NonPositiveTemperature;

  const PetscReal rhox = component(flow.dx, FlowComponent::Density);
  const PetscReal Ux = component(flow.dx, FlowComponent::U);
  const PetscReal Vx = component(flow.dx, FlowComponent::V);
  const PetscReal Wx = component(flow.dx, FlowComponent::W);
  const PetscReal Tx = component(flow.dx, FlowComponent::Temperature);
  const PetscReal rhoy = component(flow.dy, FlowComponent::Density);
  const PetscReal Uy = component(flow.dy, FlowComponent::U);
  const PetscReal Vy = component(flow.dy, FlowComponent::V);
  const PetscReal Wy = component(flow.dy, FlowComponent::W);
  const PetscReal Ty = component(flow.dy, FlowComponent::Temperature);
  const PetscReal rhoz = component(flow.dz, FlowComponent::Density);
  const PetscReal Uz = component(flow.dz, FlowComponent::U);
  const PetscReal Vz = component(flow.dz, FlowComponent::V);
  const PetscReal Wz = component(flow.dz, FlowComponent::W);
  const PetscReal Tz = component(flow.dz, FlowComponent::Temperature);

  const PetscReal Uxx = component(flow.dxx, FlowComponent::U);
  const PetscReal Vxx = component(flow.dxx, FlowComponent::V);
  const PetscReal Wxx = component(flow.dxx, FlowComponent::W);
  const PetscReal Txx = component(flow.dxx, FlowComponent::Temperature);
  const PetscReal Uyy = component(flow.dyy, FlowComponent::U);
  const PetscReal Vyy = component(flow.dyy, FlowComponent::V);
  const PetscReal Wyy = component(flow.dyy, FlowComponent::W);
  const PetscReal Tyy = component(flow.dyy, FlowComponent::Temperature);
  const PetscReal Uzz = component(flow.dzz, FlowComponent::U);
  const PetscReal Vzz = component(flow.dzz, FlowComponent::V);
  const PetscReal Wzz = component(flow.dzz, FlowComponent::W);
  const PetscReal Tzz = component(flow.dzz, FlowComponent::Temperature);
  const PetscReal Uxy = component(flow.dxy, FlowComponent::U);
  const PetscReal Vxy = component(flow.dxy, FlowComponent::V);
  const PetscReal Wxy = component(flow.dxy, FlowComponent::W);
  const PetscReal Uxz = component(flow.dxz, FlowComponent::U);
  const PetscReal Vxz = component(flow.dxz, FlowComponent::V);
  const PetscReal Wxz = component(flow.dxz, FlowComponent::W);
  const PetscReal Uyz = component(flow.dyz, FlowComponent::U);
  const PetscReal Vyz = component(flow.dyz, FlowComponent::V);
  const PetscReal Wyz = component(flow.dyz, FlowComponent::W);
  (void)Vxy;
  (void)Vxz;
  (void)Uyz;
  (void)Wxy;

  const PetscReal Pe = 1.0 / (gamma_ * mach_ * mach_);
  const PetscReal g2 = 1.0 / ((gamma_ - 1.0) * mach_ * mach_);
  const PetscReal mu =
      T * std::sqrt(T) * (1.0 + sutherlandRatio_) / (T + sutherlandRatio_);
  const PetscReal muT = mu * (1.5 / T - 1.0 / (T + sutherlandRatio_));
  const PetscReal muTT = muT * (1.5 / T - 1.0 / (T + sutherlandRatio_)) -
                         mu * (1.5 / (T * T) - 1.0 / ((T + sutherlandRatio_) *
                                                      (T + sutherlandRatio_)));
  const PetscReal mux = muT * Tx;
  const PetscReal muy = muT * Ty;
  const PetscReal muz = muT * Tz;

  coefficients = {};
  auto &G = coefficients.Gamma;
  auto &A = coefficients.A;
  auto &B = coefficients.B;
  auto &C = coefficients.C;
  auto &D = coefficients.D;
  auto &MVxx = coefficients.Vxx;
  auto &MVxy = coefficients.Vxy;
  auto &MVxz = coefficients.Vxz;
  auto &MVyy = coefficients.Vyy;
  auto &MVyz = coefficients.Vyz;
  auto &MVzz = coefficients.Vzz;

  G(0, 0) = 1.0;
  G(1, 1) = rho;
  G(2, 2) = rho;
  G(3, 3) = rho;
  G(4, 0) = -Pe * T;
  G(4, 4) = rho * g2 - Pe * rho;

  A(0, 0) = U;
  A(0, 1) = rho;
  A(1, 0) = Pe * T;
  A(1, 1) = rho * U - fourThirds * mux / reynolds_;
  A(1, 2) = -muy / reynolds_;
  A(1, 3) = -muz / reynolds_;
  A(1, 4) = Pe * rho - muT / reynolds_ *
                           (fourThirds * Ux - twoThirds * Vy - twoThirds * Wz);
  A(2, 1) = twoThirds * muy / reynolds_;
  A(2, 2) = rho * U - mux / reynolds_;
  A(2, 4) = -muT * (Uy + Vx) / reynolds_;
  A(3, 1) = twoThirds * muz / reynolds_;
  A(3, 3) = rho * U - mux / reynolds_;
  A(3, 4) = -muT * (Wx + Uz) / reynolds_;
  A(4, 0) = -Pe * U * T;
  A(4, 1) = -2.0 * mu * (fourThirds * Ux - twoThirds * Vy - twoThirds * Wz) /
            reynolds_;
  A(4, 2) = -2.0 * mu * (Uy + Vx) / reynolds_;
  A(4, 3) = -2.0 * mu * (Wx + Uz) / reynolds_;
  A(4, 4) = rho * U * g2 - rho * U * Pe -
            2.0 * Tx * muT * g2 / (reynolds_ * prandtl_);

  B(0, 0) = V;
  B(0, 2) = rho;
  B(1, 1) = rho * V - muy / reynolds_;
  B(1, 2) = twoThirds * mux / reynolds_;
  B(1, 4) = -muT * (Uy + Vx) / reynolds_;
  B(2, 0) = Pe * T;
  B(2, 1) = -mux / reynolds_;
  B(2, 2) = rho * V - fourThirds * muy / reynolds_;
  B(2, 3) = -muz / reynolds_;
  B(2, 4) = Pe * rho - muT / reynolds_ *
                           (-twoThirds * Ux + fourThirds * Vy - twoThirds * Wz);
  B(3, 2) = twoThirds * muz / reynolds_;
  B(3, 3) = rho * V - muy / reynolds_;
  B(3, 4) = -muT * (Vz + Wy) / reynolds_;
  B(4, 0) = -Pe * V * T;
  B(4, 1) = -2.0 * mu * (Uy + Vx) / reynolds_;
  B(4, 2) = -2.0 * mu * (-twoThirds * Ux + fourThirds * Vy - twoThirds * Wz) /
            reynolds_;
  B(4, 3) = -2.0 * mu * (Vz + Wy) / reynolds_;
  B(4, 4) = rho * V * g2 - rho * V * Pe -
            2.0 * Ty * muT * g2 / (reynolds_ * prandtl_);

  C(0, 0) = W;
  C(0, 3) = rho;
  C(1, 1) = rho * W - muz / reynolds_;
  C(1, 3) = twoThirds * mux / reynolds_;
  C(1, 4) = -muT * (Wx + Uz) / reynolds_;
  C(2, 2) = rho * W - muz / reynolds_;
  C(2, 3) = twoThirds * muy / reynolds_;
  C(2, 4) = -muT * (Vz + Wy) / reynolds_;
  C(3, 0) = Pe * T;
  C(3, 1) = -mux / reynolds_;
  C(3, 2) = -muy / reynolds_;
  C(3, 3) = rho * W - fourThirds * muz / reynolds_;
  C(3, 4) = Pe * rho - muT / reynolds_ *
                           (-twoThirds * Ux - twoThirds * Vy + fourThirds * Wz);
  C(4, 0) = -Pe * W * T;
  C(4, 1) = -2.0 * mu * (Wx + Uz) / reynolds_;
  C(4, 2) = -2.0 * mu * (Vz + Wy) / reynolds_;
  C(4, 3) = -2.0 * mu * (-twoThirds * Ux - twoThirds * Vy + fourThirds * Wz) /
            reynolds_;
  C(4, 4) = rho * W * g2 - rho * W * Pe -
            2.0 * Tz * muT * g2 / (reynolds_ * prandtl_);

  D(0, 0) = Ux + Vy + Wz;
  D(0, 1) = rhox;
  D(0, 2) = rhoy;
  D(0, 3) = rhoz;
  D(1, 0) = U * Ux + V * Uy + W * Uz + Pe * Tx;
  D(1, 1) = rho * Ux;
  D(1, 2) = rho * Uy;
  D(1, 3) = rho * Uz;
  D(1, 4) =
      Pe * rhox -
      (muTT * Tx * twoThirds * (2.0 * Ux - Vy - Wz) + muTT * Ty * (Uy + Vx) +
       muTT * Tz * (Wx + Uz) +
       muT * (fourThirds * Uxx + Uyy + Uzz + oneThird * Vxy + oneThird * Wxz)) /
          reynolds_;
  D(2, 0) = U * Vx + V * Vy + W * Vz + Pe * Ty;
  D(2, 1) = rho * Vx;
  D(2, 2) = rho * Vy;
  D(2, 3) = rho * Vz;
  D(2, 4) =
      Pe * rhoy -
      (muTT * Tx * (Uy + Vx) + muTT * Ty * twoThirds * (-Ux + 2.0 * Vy - Wz) +
       muTT * Tz * (Vz + Wy) +
       muT * (Vxx + fourThirds * Vyy + Vzz + oneThird * Uxy + oneThird * Wyz)) /
          reynolds_;
  D(3, 0) = U * Wx + V * Wy + W * Wz + Pe * Tz;
  D(3, 1) = rho * Wx;
  D(3, 2) = rho * Wy;
  D(3, 3) = rho * Wz;
  D(3, 4) =
      Pe * rhoz -
      (muTT * Tx * (Wx + Uz) + muTT * Ty * (Vz + Wy) +
       muTT * Tz * twoThirds * (-Ux - Vy + 2.0 * Wz) +
       muT * (Wxx + Wyy + fourThirds * Wzz + oneThird * Uxz + oneThird * Vyz)) /
          reynolds_;
  D(4, 0) = (g2 - Pe) * (U * Tx + V * Ty + W * Tz);
  D(4, 1) = rho * Tx * g2 - Pe * (rho * Tx + T * rhox);
  D(4, 2) = rho * Ty * g2 - Pe * (rho * Ty + T * rhoy);
  D(4, 3) = rho * Tz * g2 - Pe * (rho * Tz + T * rhoz);
  D(4, 4) = -Pe * (U * rhox + V * rhoy + W * rhoz) -
            (Txx + Tyy + Tzz) * muT * g2 / (reynolds_ * prandtl_) -
            (Tx * Tx + Ty * Ty + Tz * Tz) * muTT * g2 / (reynolds_ * prandtl_) -
            muT *
                (Uy * Uy + Vx * Vx + 2.0 * Uy * Vx + Uz * Uz + Wx * Wx +
                 2.0 * Uz * Wx + Vz * Vz + Wy * Wy + 2.0 * Vz * Wy) /
                reynolds_ -
            muT * fourThirds *
                (Ux * Ux + Vy * Vy + Wz * Wz - Ux * Wz - Ux * Vy - Vy * Wz) /
                reynolds_;

  MVxx(1, 1) = fourThirds * mu / reynolds_;
  MVxx(2, 2) = mu / reynolds_;
  MVxx(3, 3) = mu / reynolds_;
  MVxx(4, 4) = mu * g2 / (reynolds_ * prandtl_);
  MVyy(1, 1) = mu / reynolds_;
  MVyy(2, 2) = fourThirds * mu / reynolds_;
  MVyy(3, 3) = mu / reynolds_;
  MVyy(4, 4) = mu * g2 / (reynolds_ * prandtl_);
  MVzz(1, 1) = mu / reynolds_;
  MVzz(2, 2) = mu / reynolds_;
  MVzz(3, 3) = fourThirds * mu / reynolds_;
  MVzz(4, 4) = mu * g2 / (reynolds_ * prandtl_);
  MVxy(1, 2) = oneThird * mu / reynolds_;
  MVxy(2, 1) = oneThird * mu / reynolds_;
  MVxz(1, 3) = oneThird * mu / reynolds_;
  MVxz(3, 1) = oneThird * mu / reynolds_;
  MVyz(2, 3) = oneThird * mu / reynolds_;
  MVyz(3, 2) = oneThird * mu / reynolds_;

  if (!coefficients.finite())
    return CoefficientStatus::NonFiniteOutput;
  return CoefficientStatus::Success;
}
