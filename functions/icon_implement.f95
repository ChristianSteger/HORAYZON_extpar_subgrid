! Description: Compute sun position and correction of SW_dir according to
!							 ICON (icon-nwp/src/atm_phy_schemes/mo_radiation.f90)
!
! Compilation:
! f2py -c --fcompiler=gnu95 icon_implement.f95 -m icon_implement
! f2py -c --fcompiler=gnu95 --opt=-O3 icon_implement.f95 -m icon_implement
!
! Author: Christian Steger, September 2024

SUBROUTINE interpolate(horizon, ztheta_sun, zphi_sun, fcor_sun)

  IMPLICIT NONE

  ! 8 byte real kind parameter
  INTEGER, PARAMETER :: dp = selected_real_kind(13)
  ! 4 byte real kind parameter
  INTEGER, PARAMETER :: sp = selected_real_kind(6)
  ! INTEGER, PARAMETER :: wp = sp  ! single precision
  INTEGER, PARAMETER :: wp = dp  ! double precision

  REAL(KIND=wp), DIMENSION(192) :: horizon ! start elevation angle and f_cor-values [degree, -]
  REAL(KIND=wp) :: zphi_sun   ! sun azimuth angle [rad]
  REAL(KIND=wp) :: ztheta_sun ! sun elevation angle [rad]
  REAL(KIND=wp) :: fcor_sun    ! interpolated f_cor_value for sun position [-]

  INTEGER :: nhori
  REAL(KIND=wp) :: pi, rad2deg
  INTEGER :: k, ii
  INTEGER :: ind_hori
  REAL(KIND=wp) :: zihor
  REAL(KIND=wp) :: eta
  INTEGER :: num_elem
  REAL(KIND=wp) :: elev_start, elev_end
  REAL(KIND=wp) :: pos_norm
  REAL(KIND=wp) :: fcor_left, fcor_right
  INTEGER :: ind_low
  REAL(KIND=wp) :: elev_low, elev_up, weight_low

  ! Constant values
  nhori = 24
  eta = 2.1_wp
  num_elem = 8
  elev_end = 90.0_wp
  
  !f2py threadsafe
  !f2py intent(in) horizon
  !f2py intent(in) zphi_sun
  !f2py intent(in) ztheta_sun
  !f2py intent(out) fcor_sun

  ! Constants
  pi = 4.0_wp * ATAN(1.0_wp) ! -> where defined in ICON?
  rad2deg = 180.0_wp / pi

  ! Compute azimuth indices
  zihor = REAL(INT(360.0_wp/nhori),wp) ! 15 degree
  ii = MIN(nhori-1, INT(rad2deg*zphi_sun/zihor)) ! index starting with 0!
  k = MOD(ii+1,nhori)                            ! index starting with 0!
  PRINT *, "ii =", ii, " k =", k

  ! Left f_cor-value
  ind_hori = ii * num_elem + 1
  PRINT *, "ind_hori =", ind_hori
  elev_start = horizon(ind_hori)
  PRINT *, "elev_start =", elev_start
  pos_norm = (ztheta_sun*rad2deg - elev_start) / (elev_end - elev_start)
  PRINT *, "pos_norm =", pos_norm
  IF (pos_norm <= 0.0) THEN
    fcor_left = 0.0_wp
  ELSE
    ind_low = INT((num_elem - 2) * pos_norm ** (1.0 / eta)) + 1
    PRINT *, "ind_low =", ind_low
    IF (ind_low >= (num_elem - 1)) THEN
      fcor_left = 1.0_wp
    ELSE
      elev_low = elev_start + (elev_end - elev_start) * (REAL(ind_low-1,wp) / REAL(num_elem - 2,wp)) ** eta
      PRINT *, "elev_low =", elev_low
      elev_up = elev_start + (elev_end - elev_start) * (REAL(ind_low,wp) / REAL(num_elem - 2,wp)) ** eta
      PRINT *, "elev_up =", elev_up
      weight_low = (elev_up - ztheta_sun*rad2deg) / (elev_up - elev_low)
      PRINT *, "weight_low =", weight_low
      fcor_left = horizon(ind_hori+ind_low) * weight_low + horizon(ind_hori+ind_low+1) * (1.0_wp - weight_low)
      PRINT *, "fcor_left =", fcor_left
    ENDIF
  ENDIF

  ! Right f_cor-value
  ind_hori = k * num_elem + 1
  PRINT *, "ind_hori =", ind_hori
  elev_start = horizon(ind_hori)
  PRINT *, "elev_start =", elev_start
  pos_norm = (ztheta_sun*rad2deg - elev_start) / (elev_end - elev_start)
  PRINT *, "pos_norm =", pos_norm
  IF (pos_norm <= 0.0) THEN
    fcor_right = 0.0_wp
  ELSE
    ind_low = INT((num_elem - 2) * pos_norm ** (1.0 / eta)) + 1
    PRINT *, "ind_low =", ind_low
    IF (ind_low >= (num_elem - 1)) THEN
      fcor_right = 1.0_wp
    ELSE
      elev_low = elev_start + (elev_end - elev_start) * (REAL(ind_low-1,wp) / REAL(num_elem - 2,wp)) ** eta
      PRINT *, "elev_low =", elev_low
      elev_up = elev_start + (elev_end - elev_start) * (REAL(ind_low,wp) / REAL(num_elem - 2,wp)) ** eta
      PRINT *, "elev_up =", elev_up
      weight_low = (elev_up - ztheta_sun*rad2deg) / (elev_up - elev_low)
      PRINT *, "weight_low =", weight_low
      fcor_right = horizon(ind_hori+ind_low) * weight_low + horizon(ind_hori+ind_low+1) * (1.0_wp - weight_low)
      PRINT *, "fcor_left =", fcor_left
    ENDIF
  ENDIF

  ! Interpolate f_cor-value at azimuth angle
  fcor_sun = (fcor_right*(rad2deg*zphi_sun-zihor*ii) + fcor_left*(zihor*(ii+1)-rad2deg*zphi_sun))/ zihor

END SUBROUTINE interpolate
