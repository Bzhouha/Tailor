!> @file generate_lns_transform_gold.f90
!! @brief Generate trusted Fourier and curvilinear transformation fixtures.
program generate_lns_transform_gold
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  complex(dp) :: physical(5,5,11), fourier(5,5,7)
  complex(dp) :: curvilinear(5,5,7), alpha
  real(dp) :: metrics(10)
  integer :: point, matrix, row, column, output_unit
  character(len=1024) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) > 0) then
    open(newunit=output_unit, file=trim(output_path), status='replace', &
         action='write', form='formatted')
  else
    output_unit = 6
  end if

  do point = 1, 2
    call make_physical(point, physical)
    call make_parameters(point, alpha, metrics)
    call streamwise_fourier(physical, alpha, fourier)
    call curvilinear_transform(fourier, metrics, curvilinear)

    do matrix = 1, 7
      do row = 1, 5
        do column = 1, 5
          write(output_unit,'(2ES25.17E3)') &
              real(fourier(row,column,matrix)), &
              aimag(fourier(row,column,matrix))
        end do
      end do
    end do
    do matrix = 1, 7
      do row = 1, 5
        do column = 1, 5
          write(output_unit,'(2ES25.17E3)') &
              real(curvilinear(row,column,matrix)), &
              aimag(curvilinear(row,column,matrix))
        end do
      end do
    end do
  end do

  if (output_unit /= 6) close(output_unit)

contains

!> @brief Construct deterministic physical-space coefficient matrices.
!! @param[in] point Fixture point identifier.
!! @param[out] physical Eleven physical coefficient matrices.
subroutine make_physical(point, physical)
  implicit none
  integer, intent(in) :: point
  complex(dp), intent(out) :: physical(5,5,11)
  integer :: matrix, row, column
  real(dp) :: value

  do matrix = 1, 11
    do row = 1, 5
      do column = 1, 5
        value = 0.13_dp*matrix + 0.017_dp*row + 0.003_dp*column + &
                0.0007_dp*point
        if (mod(matrix + row + column + point, 2) == 0) value = -value
        physical(row,column,matrix) = cmplx(value, 0.0_dp, dp)
      end do
    end do
  end do
end subroutine make_physical

!> @brief Construct a complex wave number and nontrivial metric coefficients.
!! @param[in] point Fixture point identifier.
!! @param[out] alpha Complex streamwise wave number.
!! @param[out] metrics Ten first- and second-order metric coefficients.
subroutine make_parameters(point, alpha, metrics)
  implicit none
  integer, intent(in) :: point
  complex(dp), intent(out) :: alpha
  real(dp), intent(out) :: metrics(10)

  if (point == 1) then
    alpha = cmplx(2.43_dp, 0.17_dp, dp)
    metrics = [0.81_dp, -0.23_dp, 0.19_dp, 1.07_dp, &
               -0.031_dp, 0.024_dp, -0.017_dp, &
               0.028_dp, -0.022_dp, 0.013_dp]
  else
    alpha = cmplx(-0.64_dp, 0.29_dp, dp)
    metrics = [1.14_dp, 0.37_dp, -0.26_dp, 0.72_dp, &
               0.041_dp, -0.036_dp, 0.025_dp, &
               -0.033_dp, 0.019_dp, -0.027_dp]
  end if
end subroutine make_parameters

!> @brief Apply the streamwise Fourier substitution to physical coefficients.
!! @param[in] physical Eleven physical coefficient matrices.
!! @param[in] alpha Complex streamwise wave number.
!! @param[out] fourier Seven Fourier-space coefficient matrices.
subroutine streamwise_fourier(physical, alpha, fourier)
  implicit none
  complex(dp), intent(in) :: physical(5,5,11), alpha
  complex(dp), intent(out) :: fourier(5,5,7)
  complex(dp), parameter :: ci = cmplx(0.0_dp, 1.0_dp, dp)

  fourier = cmplx(0.0_dp, 0.0_dp, dp)
  fourier(:,:,1) = physical(:,:,1)
  fourier(:,:,2) = physical(:,:,5) + ci*alpha*physical(:,:,2) + &
                   alpha*alpha*physical(:,:,6)
  fourier(:,:,3) = physical(:,:,3) - ci*alpha*physical(:,:,7)
  fourier(:,:,4) = physical(:,:,4) - ci*alpha*physical(:,:,8)
  fourier(:,:,5) = physical(:,:,9)
  fourier(:,:,6) = physical(:,:,10)
  fourier(:,:,7) = physical(:,:,11)
end subroutine streamwise_fourier

!> @brief Transform Fourier-space coefficients to computational coordinates.
!! @param[in] fourier Seven Fourier-space coefficient matrices.
!! @param[in] metric Ten curvilinear metric coefficients.
!! @param[out] curvilinear Seven computational-space coefficient matrices.
subroutine curvilinear_transform(fourier, metric, curvilinear)
  implicit none
  complex(dp), intent(in) :: fourier(5,5,7)
  real(dp), intent(in) :: metric(10)
  complex(dp), intent(out) :: curvilinear(5,5,7)
  real(dp) :: xi_y, xi_z, eta_y, eta_z
  real(dp) :: xi_yy, xi_zz, xi_yz, eta_yy, eta_zz, eta_yz

  xi_y=metric(1); xi_z=metric(2)
  eta_y=metric(3); eta_z=metric(4)
  xi_yy=metric(5); xi_zz=metric(6); xi_yz=metric(7)
  eta_yy=metric(8); eta_zz=metric(9); eta_yz=metric(10)

  curvilinear = cmplx(0.0_dp, 0.0_dp, dp)
  curvilinear(:,:,1) = fourier(:,:,1)
  curvilinear(:,:,2) = fourier(:,:,2)
  curvilinear(:,:,3) = xi_y*fourier(:,:,3) + xi_z*fourier(:,:,4) - &
                       xi_yy*fourier(:,:,5) - xi_zz*fourier(:,:,7) - &
                       xi_yz*fourier(:,:,6)
  curvilinear(:,:,4) = eta_y*fourier(:,:,3) + eta_z*fourier(:,:,4) - &
                       eta_yy*fourier(:,:,5) - eta_zz*fourier(:,:,7) - &
                       eta_yz*fourier(:,:,6)
  curvilinear(:,:,5) = xi_y*xi_y*fourier(:,:,5) + &
                       xi_z*xi_z*fourier(:,:,7) + &
                       xi_y*xi_z*fourier(:,:,6)
  curvilinear(:,:,6) = 2.0_dp*xi_y*eta_y*fourier(:,:,5) + &
                       2.0_dp*xi_z*eta_z*fourier(:,:,7) + &
                       (xi_y*eta_z + eta_y*xi_z)*fourier(:,:,6)
  curvilinear(:,:,7) = eta_y*eta_y*fourier(:,:,5) + &
                       eta_z*eta_z*fourier(:,:,7) + &
                       eta_y*eta_z*fourier(:,:,6)
end subroutine curvilinear_transform

end program generate_lns_transform_gold
