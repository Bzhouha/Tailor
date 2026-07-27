#include "CurvilinearTransform.hpp"

#include <petscmath.h>

#include <array>

namespace {

constexpr std::array<MetricComponent,
                     static_cast<std::size_t>(metricComponentCount)>
    metricComponents = {
        MetricComponent::XiY,   MetricComponent::XiZ,   MetricComponent::EtaY,
        MetricComponent::EtaZ,  MetricComponent::XiYY,  MetricComponent::XiZZ,
        MetricComponent::XiYZ,  MetricComponent::EtaYY, MetricComponent::EtaZZ,
        MetricComponent::EtaYZ,
};

} // namespace

bool MetricPoint::finite() const noexcept {
  const std::array<PetscReal, metricComponentCount> values = {
      xiY, xiZ, etaY, etaZ, xiYY, xiZZ, xiYZ, etaYY, etaZZ, etaYZ,
  };
  for (const PetscReal value : values) {
    if (PetscIsInfOrNanReal(value))
      return false;
  }
  return true;
}

CoefficientStatus makeMetricPoint(const PetscScalar *metrics,
                                  MetricPoint &point) noexcept {
  if (!metrics)
    return CoefficientStatus::NonFiniteInput;

  std::array<PetscReal, metricComponentCount> values{};
  for (PetscInt component = 0; component < metricComponentCount; ++component) {
    const PetscScalar value = metrics[metricIndex(
        metricComponents[static_cast<std::size_t>(component)])];
    if (PetscIsInfOrNanScalar(value) || PetscImaginaryPart(value) != 0.0)
      return CoefficientStatus::NonFiniteInput;
    values[static_cast<std::size_t>(component)] = PetscRealPart(value);
  }

  point = {
      values[0], values[1], values[2], values[3], values[4],
      values[5], values[6], values[7], values[8], values[9],
  };
  return point.finite() ? CoefficientStatus::Success
                        : CoefficientStatus::NonFiniteInput;
}

bool CurvilinearLNSCoefficients::finite() const noexcept {
  return Gamma.finite() && K0.finite() && Kxi.finite() && Keta.finite() &&
         Vxixi.finite() && Vxieta.finite() && Vetaeta.finite();
}

CoefficientStatus CurvilinearTransform::apply(
    const FourierLNSCoefficients &fourier, const MetricPoint &metrics,
    CurvilinearLNSCoefficients &curvilinear) const noexcept {
  if (!fourier.finite() || !metrics.finite())
    return CoefficientStatus::NonFiniteInput;

  curvilinear = {};
  for (PetscInt row = 0; row < flowComponentCount; ++row) {
    for (PetscInt column = 0; column < flowComponentCount; ++column) {
      const PetscScalar Vyy = fourier.Vyy(row, column);
      const PetscScalar Vyz = fourier.Vyz(row, column);
      const PetscScalar Vzz = fourier.Vzz(row, column);

      curvilinear.Gamma(row, column) = fourier.Gamma(row, column);
      curvilinear.K0(row, column) = fourier.K0(row, column);
      curvilinear.Kxi(row, column) = metrics.xiY * fourier.Ky(row, column) +
                                     metrics.xiZ * fourier.Kz(row, column) -
                                     metrics.xiYY * Vyy - metrics.xiZZ * Vzz -
                                     metrics.xiYZ * Vyz;
      curvilinear.Keta(row, column) = metrics.etaY * fourier.Ky(row, column) +
                                      metrics.etaZ * fourier.Kz(row, column) -
                                      metrics.etaYY * Vyy -
                                      metrics.etaZZ * Vzz - metrics.etaYZ * Vyz;
      curvilinear.Vxixi(row, column) = metrics.xiY * metrics.xiY * Vyy +
                                       metrics.xiZ * metrics.xiZ * Vzz +
                                       metrics.xiY * metrics.xiZ * Vyz;
      curvilinear.Vetaeta(row, column) = metrics.etaY * metrics.etaY * Vyy +
                                         metrics.etaZ * metrics.etaZ * Vzz +
                                         metrics.etaY * metrics.etaZ * Vyz;
      curvilinear.Vxieta(row, column) =
          2.0 * metrics.xiY * metrics.etaY * Vyy +
          2.0 * metrics.xiZ * metrics.etaZ * Vzz +
          (metrics.xiY * metrics.etaZ + metrics.etaY * metrics.xiZ) * Vyz;
    }
  }

  if (!curvilinear.finite())
    return CoefficientStatus::NonFiniteOutput;
  return CoefficientStatus::Success;
}
