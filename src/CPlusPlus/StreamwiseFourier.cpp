#include "StreamwiseFourier.hpp"

#include <petscmath.h>

bool FourierLNSCoefficients::finite() const noexcept {
  return Gamma.finite() && K0.finite() && Ky.finite() && Kz.finite() &&
         Vyy.finite() && Vyz.finite() && Vzz.finite();
}

StreamwiseFourier::StreamwiseFourier(PetscScalar alpha) : alpha_(alpha) {}

CoefficientStatus
StreamwiseFourier::apply(const PhysicalLNSCoefficients &physical,
                         FourierLNSCoefficients &fourier) const noexcept {
  if (PetscIsInfOrNanScalar(alpha_) || !physical.finite())
    return CoefficientStatus::NonFiniteInput;

  fourier = {};
  const PetscScalar imaginaryAlpha = PETSC_i * alpha_;
  const PetscScalar alphaSquared = alpha_ * alpha_;
  for (PetscInt row = 0; row < flowComponentCount; ++row) {
    for (PetscInt column = 0; column < flowComponentCount; ++column) {
      fourier.Gamma(row, column) = physical.Gamma(row, column);
      fourier.K0(row, column) = physical.D(row, column) +
                                imaginaryAlpha * physical.A(row, column) +
                                alphaSquared * physical.Vxx(row, column);
      fourier.Ky(row, column) =
          physical.B(row, column) - imaginaryAlpha * physical.Vxy(row, column);
      fourier.Kz(row, column) =
          physical.C(row, column) - imaginaryAlpha * physical.Vxz(row, column);
      fourier.Vyy(row, column) = physical.Vyy(row, column);
      fourier.Vyz(row, column) = physical.Vyz(row, column);
      fourier.Vzz(row, column) = physical.Vzz(row, column);
    }
  }

  if (!fourier.finite())
    return CoefficientStatus::NonFiniteOutput;
  return CoefficientStatus::Success;
}
