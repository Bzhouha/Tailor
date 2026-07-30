!> @file generate_lns_coefficients_gold.f90
!! @brief Generate trusted LNS coefficient fixtures from the reference formulas.
program generate_lns_coefficients_gold
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  real(dp) :: points(5,10,2), matrices(5,5,11)
  integer :: point, matrix, row, column, output_unit
  character(len=1024) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) > 0) then
    open(newunit=output_unit, file=trim(output_path), status='replace', &
         action='write', form='formatted')
  else
    output_unit = 6
  end if

  points = 0.0_dp
  points(:,1,1) = [1.2_dp, 2.3_dp, 0.14_dp, -0.08_dp, 1.4_dp]
  points(:,2,1) = [0.01_dp, 0.02_dp, -0.03_dp, 0.04_dp, 0.05_dp]
  points(:,3,1) = [-0.02_dp, 0.07_dp, 0.04_dp, -0.05_dp, 0.06_dp]
  points(:,4,1) = [0.03_dp, -0.02_dp, 0.08_dp, 0.09_dp, -0.04_dp]
  points(:,5,1) = [0.004_dp, -0.006_dp, 0.007_dp, -0.008_dp, 0.009_dp]
  points(:,6,1) = [-0.005_dp, 0.011_dp, -0.012_dp, 0.013_dp, -0.014_dp]
  points(:,7,1) = [0.006_dp, -0.015_dp, 0.016_dp, -0.017_dp, 0.018_dp]
  points(:,8,1) = [-0.007_dp, 0.019_dp, -0.020_dp, 0.021_dp, -0.022_dp]
  points(:,9,1) = [0.008_dp, -0.023_dp, 0.024_dp, -0.025_dp, 0.026_dp]
  points(:,10,1) = [-0.009_dp, 0.027_dp, -0.028_dp, 0.029_dp, -0.030_dp]

  points(:,1,2) = [0.95_dp, 3.1_dp, -0.12_dp, 0.06_dp, 1.15_dp]
  points(:,3,2) = [0.015_dp, -0.045_dp, 0.025_dp, 0.035_dp, -0.055_dp]
  points(:,4,2) = [-0.012_dp, 0.038_dp, -0.028_dp, 0.042_dp, 0.048_dp]
  points(:,6,2) = [0.003_dp, -0.009_dp, 0.008_dp, -0.007_dp, 0.011_dp]
  points(:,7,2) = [-0.004_dp, 0.010_dp, -0.012_dp, 0.014_dp, -0.013_dp]
  points(:,10,2) = [0.005_dp, -0.016_dp, 0.017_dp, -0.018_dp, 0.019_dp]

  do point = 1, 2
    call get_unadorned(points(:,:,point), matrices)
    do matrix = 1, 11
      do row = 1, 5
        do column = 1, 5
          write(output_unit,'(ES25.17E3)') matrices(row,column,matrix)
        end do
      end do
    end do
  end do
  if (output_unit /= 6) close(output_unit)

contains

!> @brief Evaluate all eleven physical-space LNS coefficient matrices.
!! @param[in] q Base-flow state and derivatives at one point.
!! @param[out] matrices Gamma, A, B, C, D, and six viscous matrices.
subroutine get_unadorned(q, matrices)
  implicit none
  real(dp), intent(in) :: q(5,10)
  real(dp), intent(out) :: matrices(5,5,11)
  real(dp), parameter :: Re=32500.0_dp, Ma=6.0_dp, Pr=0.72_dp
  real(dp), parameter :: gamma=1.4_dp, Te=63.334_dp, C1=110.4_dp
  real(dp), parameter :: d1d3=1.0_dp/3.0_dp
  real(dp), parameter :: d2d3=2.0_dp/3.0_dp
  real(dp), parameter :: d4d3=4.0_dp/3.0_dp
  real(dp) :: G(5,5), A(5,5), B(5,5), C(5,5), D(5,5)
  real(dp) :: Vxxm(5,5), Vyym(5,5), Vzzm(5,5)
  real(dp) :: Vxym(5,5), Vxzm(5,5), Vyzm(5,5)
  real(dp) :: Pe, g2, cm, mu, muT, muTT, mux, muy, muz
  real(dp) :: rho,U,V,W,T,rhox,Ux,Vx,Wx,Tx,rhoy,Uy,Vy,Wy,Ty
  real(dp) :: rhoz,Uz,Vz,Wz,Tz,Uxx,nVxx,Wxx,Txx,Uyy,nVyy,Wyy,Tyy
  real(dp) :: Uzz,nVzz,Wzz,Tzz,Uxy,nVxy,Wxy,Uxz,nVxz,Wxz
  real(dp) :: Uyz,nVyz,Wyz

  G=0; A=0; B=0; C=0; D=0
  Vxxm=0; Vyym=0; Vzzm=0; Vxym=0; Vxzm=0; Vyzm=0
  rho=q(1,1); U=q(2,1); V=q(3,1); W=q(4,1); T=q(5,1)
  rhox=q(1,2); Ux=q(2,2); Vx=q(3,2); Wx=q(4,2); Tx=q(5,2)
  rhoy=q(1,3); Uy=q(2,3); Vy=q(3,3); Wy=q(4,3); Ty=q(5,3)
  rhoz=q(1,4); Uz=q(2,4); Vz=q(3,4); Wz=q(4,4); Tz=q(5,4)
  Uxx=q(2,5); nVxx=q(3,5); Wxx=q(4,5); Txx=q(5,5)
  Uyy=q(2,6); nVyy=q(3,6); Wyy=q(4,6); Tyy=q(5,6)
  Uzz=q(2,7); nVzz=q(3,7); Wzz=q(4,7); Tzz=q(5,7)
  Uxy=q(2,8); nVxy=q(3,8); Wxy=q(4,8)
  Uxz=q(2,9); nVxz=q(3,9); Wxz=q(4,9)
  Uyz=q(2,10); nVyz=q(3,10); Wyz=q(4,10)

  Pe=1.0_dp/(gamma*Ma*Ma)
  g2=1.0_dp/((gamma-1.0_dp)*Ma*Ma)
  cm=C1/Te
  mu=T*sqrt(T)*(1.0_dp+cm)/(T+cm)
  muT=mu*(1.5_dp/T-1.0_dp/(T+cm))
  muTT=muT*(1.5_dp/T-1.0_dp/(T+cm))-&
       mu*(1.5_dp/T**2-1.0_dp/(T+cm)**2)
  mux=muT*Tx; muy=muT*Ty; muz=muT*Tz

  G(1,1)=1; G(2,2)=rho; G(3,3)=rho; G(4,4)=rho
  G(5,1)=-Pe*T; G(5,5)=rho*g2-Pe*rho

  A(1,1)=U; A(1,2)=rho; A(2,1)=Pe*T
  A(2,2)=rho*U-d4d3*mux/Re
  A(2,3)=-muy/Re; A(2,4)=-muz/Re
  A(2,5)=Pe*rho-muT/Re*(d4d3*Ux-d2d3*Vy-d2d3*Wz)
  A(3,2)=d2d3*muy/Re; A(3,3)=rho*U-mux/Re
  A(3,5)=-muT*(Uy+Vx)/Re
  A(4,2)=d2d3*muz/Re; A(4,4)=rho*U-mux/Re
  A(4,5)=-muT*(Wx+Uz)/Re
  A(5,1)=-Pe*U*T
  A(5,2)=-2.0_dp*mu*(d4d3*Ux-d2d3*Vy-d2d3*Wz)/Re
  A(5,3)=-2.0_dp*mu*(Uy+Vx)/Re
  A(5,4)=-2.0_dp*mu*(Wx+Uz)/Re
  A(5,5)=rho*U*g2-rho*U*Pe-2.0_dp*Tx*muT/Re/Pr*g2

  B(1,1)=V; B(1,3)=rho
  B(2,2)=rho*V-muy/Re; B(2,3)=d2d3*mux/Re
  B(2,5)=-muT*(Uy+Vx)/Re
  B(3,1)=Pe*T; B(3,2)=-mux/Re
  B(3,3)=rho*V-d4d3*muy/Re; B(3,4)=-muz/Re
  B(3,5)=Pe*rho-muT*(-d2d3*Ux+d4d3*Vy-d2d3*Wz)/Re
  B(4,3)=d2d3*muz/Re; B(4,4)=rho*V-muy/Re
  B(4,5)=-muT*(Vz+Wy)/Re
  B(5,1)=-Pe*V*T; B(5,2)=-2.0_dp*mu*(Uy+Vx)/Re
  B(5,3)=-2.0_dp*mu*(-d2d3*Ux+d4d3*Vy-d2d3*Wz)/Re
  B(5,4)=-2.0_dp*mu*(Vz+Wy)/Re
  B(5,5)=rho*V*g2-rho*V*Pe-2.0_dp*Ty*muT*g2/Re/Pr

  C(1,1)=W; C(1,4)=rho
  C(2,2)=rho*W-muz/Re; C(2,4)=d2d3*mux/Re
  C(2,5)=-muT*(Wx+Uz)/Re
  C(3,3)=rho*W-muz/Re; C(3,4)=d2d3*muy/Re
  C(3,5)=-muT*(Vz+Wy)/Re
  C(4,1)=Pe*T; C(4,2)=-mux/Re; C(4,3)=-muy/Re
  C(4,4)=rho*W-d4d3*muz/Re
  C(4,5)=Pe*rho-muT*(-d2d3*Ux-d2d3*Vy+d4d3*Wz)/Re
  C(5,1)=-Pe*W*T; C(5,2)=-2.0_dp*mu*(Wx+Uz)/Re
  C(5,3)=-2.0_dp*mu*(Vz+Wy)/Re
  C(5,4)=-2.0_dp*mu*(-d2d3*Ux-d2d3*Vy+d4d3*Wz)/Re
  C(5,5)=rho*W*g2-rho*W*Pe-2.0_dp*Tz*muT*g2/Re/Pr

  D(1,1)=Ux+Vy+Wz; D(1,2)=rhox; D(1,3)=rhoy; D(1,4)=rhoz
  D(2,1)=U*Ux+V*Uy+W*Uz+Pe*Tx
  D(2,2)=rho*Ux; D(2,3)=rho*Uy; D(2,4)=rho*Uz
  D(2,5)=Pe*rhox-(muTT*Tx*d2d3*(2.0_dp*Ux-Vy-Wz)+&
       muTT*Ty*(Uy+Vx)+muTT*Tz*(Wx+Uz)+&
       muT*(d4d3*Uxx+Uyy+Uzz+d1d3*nVxy+d1d3*Wxz))/Re
  D(3,1)=U*Vx+V*Vy+W*Vz+Pe*Ty
  D(3,2)=rho*Vx; D(3,3)=rho*Vy; D(3,4)=rho*Vz
  D(3,5)=Pe*rhoy-(muTT*Tx*(Uy+Vx)+&
       muTT*Ty*d2d3*(-Ux+2.0_dp*Vy-Wz)+muTT*Tz*(Vz+Wy)+&
       muT*(nVxx+d4d3*nVyy+nVzz+d1d3*Uxy+d1d3*Wyz))/Re
  D(4,1)=U*Wx+V*Wy+W*Wz+Pe*Tz
  D(4,2)=rho*Wx; D(4,3)=rho*Wy; D(4,4)=rho*Wz
  D(4,5)=Pe*rhoz-(muTT*Tx*(Wx+Uz)+muTT*Ty*(Vz+Wy)+&
       muTT*Tz*d2d3*(-Ux-Vy+2.0_dp*Wz)+&
       muT*(Wxx+Wyy+d4d3*Wzz+d1d3*Uxz+d1d3*nVyz))/Re
  D(5,1)=(g2-Pe)*(U*Tx+V*Ty+W*Tz)
  D(5,2)=rho*Tx*g2-Pe*(rho*Tx+T*rhox)
  D(5,3)=rho*Ty*g2-Pe*(rho*Ty+T*rhoy)
  D(5,4)=rho*Tz*g2-Pe*(rho*Tz+T*rhoz)
  D(5,5)=-Pe*(U*rhox+V*rhoy+W*rhoz)-&
       (Txx+Tyy+Tzz)*muT*g2/Re/Pr-&
       (Tx*Tx+Ty*Ty+Tz*Tz)*muTT*g2/Re/Pr-&
       muT*(Uy*Uy+Vx*Vx+2.0_dp*Uy*Vx+Uz*Uz+Wx*Wx+&
       2.0_dp*Uz*Wx+Vz*Vz+Wy*Wy+2.0_dp*Vz*Wy)/Re-&
       muT*d4d3*(Ux*Ux+Vy*Vy+Wz*Wz-Ux*Wz-Ux*Vy-Vy*Wz)/Re

  Vxxm(2,2)=d4d3*mu/Re; Vxxm(3,3)=mu/Re
  Vxxm(4,4)=mu/Re; Vxxm(5,5)=mu*g2/Re/Pr
  Vyym(2,2)=mu/Re; Vyym(3,3)=d4d3*mu/Re
  Vyym(4,4)=mu/Re; Vyym(5,5)=mu*g2/Re/Pr
  Vzzm(2,2)=mu/Re; Vzzm(3,3)=mu/Re
  Vzzm(4,4)=d4d3*mu/Re; Vzzm(5,5)=mu*g2/Re/Pr
  Vxym(2,3)=d1d3*mu/Re; Vxym(3,2)=d1d3*mu/Re
  Vxzm(2,4)=d1d3*mu/Re; Vxzm(4,2)=d1d3*mu/Re
  Vyzm(3,4)=d1d3*mu/Re; Vyzm(4,3)=d1d3*mu/Re

  matrices(:,:,1)=G; matrices(:,:,2)=A; matrices(:,:,3)=B
  matrices(:,:,4)=C; matrices(:,:,5)=D; matrices(:,:,6)=Vxxm
  matrices(:,:,7)=Vxym; matrices(:,:,8)=Vxzm
  matrices(:,:,9)=Vyym; matrices(:,:,10)=Vyzm
  matrices(:,:,11)=Vzzm
end subroutine get_unadorned

end program generate_lns_coefficients_gold
