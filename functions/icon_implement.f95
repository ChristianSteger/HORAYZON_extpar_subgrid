! Description: Compute sun position and correction of SW_dir according to
!							 ICON (icon-nwp/src/atm_phy_schemes/mo_radiation.f90)
!
! Compilation:
! f2py -c --fcompiler=gnu95 icon_implement.f95 -m icon_implement
! f2py -c --fcompiler=gnu95 --opt=-O3 icon_implement.f95 -m icon_implement
!
! Author: Christian Steger, September 2024

SUBROUTINE interpolate_fcor(horizon, ztheta_sun, zphi_sun, fcor_sun)

  IMPLICIT NONE

  ! 8 byte real kind parameter
  INTEGER, PARAMETER :: dp = selected_real_kind(13)
  ! 4 byte real kind parameter
  INTEGER, PARAMETER :: sp = selected_real_kind(6)
  ! INTEGER, PARAMETER :: wp = sp  ! single precision
  INTEGER, PARAMETER :: wp = dp  ! double precision

  REAL(wp), DIMENSION(192) :: horizon ! start elevation angle and f_cor-values [degree, -]
  REAL(wp) :: zphi_sun   ! sun azimuth angle [rad]
  REAL(wp) :: ztheta_sun ! sun elevation angle [rad]
  REAL(wp) :: fcor_sun    ! interpolated f_cor_value for sun position [-] (new variable)

  ! Variables already defined in ICON
  INTEGER :: k, ii
  REAL(wp) :: pi, rad2deg
  INTEGER :: nhori ! currently: 24, new: 24 * 8 = 192 !!!
  REAL(wp) :: zihor

  ! New variables
  INTEGER :: num_azim
  INTEGER :: ind_hori
  REAL(wp) :: eta
  INTEGER :: num_elem
  REAL(wp) :: elev_start, elev_end
  REAL(wp) :: pos_norm
  REAL(wp) :: fcor_left, fcor_right
  INTEGER :: ind_low
  REAL(wp) :: elev_low, elev_up, weight_low

  ! Constant values
  num_azim = 24
  num_elem = 8
  nhori = num_azim * num_elem ! 192
  eta = 2.1_wp
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
  zihor = REAL(INT(360.0_wp/num_azim),wp) ! 15 degree
  ii = MIN(num_azim-1, INT(rad2deg*zphi_sun/zihor)) ! index starting with 0!
  k = MOD(ii+1,num_azim)                            ! index starting with 0!

  ! Left f_cor-value
  ind_hori = ii * num_elem + 1
  elev_start = horizon(ind_hori)
  pos_norm = (ztheta_sun*rad2deg - elev_start) / (elev_end - elev_start)
  IF (pos_norm <= 0.0) THEN
    fcor_left = 0.0_wp
  ELSE
    ind_low = INT((num_elem - 2) * pos_norm ** (1.0 / eta)) + 1
    IF (ind_low >= (num_elem - 1)) THEN
      fcor_left = 1.0_wp
    ELSE
      elev_low = elev_start + (elev_end - elev_start) * (REAL(ind_low-1,wp) / REAL(num_elem - 2,wp)) ** eta
      elev_up = elev_start + (elev_end - elev_start) * (REAL(ind_low,wp) / REAL(num_elem - 2,wp)) ** eta
      weight_low = (elev_up - ztheta_sun*rad2deg) / (elev_up - elev_low)
      fcor_left = horizon(ind_hori+ind_low) * weight_low + horizon(ind_hori+ind_low+1) * (1.0_wp - weight_low)
    ENDIF
  ENDIF

  ! Right f_cor-value
  ind_hori = k * num_elem + 1
  elev_start = horizon(ind_hori)
  pos_norm = (ztheta_sun*rad2deg - elev_start) / (elev_end - elev_start)
  IF (pos_norm <= 0.0) THEN
    fcor_right = 0.0_wp
  ELSE
    ind_low = INT((num_elem - 2) * pos_norm ** (1.0 / eta)) + 1
    IF (ind_low >= (num_elem - 1)) THEN
      fcor_right = 1.0_wp
    ELSE
      elev_low = elev_start + (elev_end - elev_start) * (REAL(ind_low-1,wp) / REAL(num_elem - 2,wp)) ** eta
      elev_up = elev_start + (elev_end - elev_start) * (REAL(ind_low,wp) / REAL(num_elem - 2,wp)) ** eta
      weight_low = (elev_up - ztheta_sun*rad2deg) / (elev_up - elev_low)
      fcor_right = horizon(ind_hori+ind_low) * weight_low + horizon(ind_hori+ind_low+1) * (1.0_wp - weight_low)
    ENDIF
  ENDIF

  ! Interpolate f_cor-value at azimuth angle
  fcor_sun = (fcor_right*(rad2deg*zphi_sun-zihor*ii) + fcor_left*(zihor*(ii+1)-rad2deg*zphi_sun))/ zihor

END SUBROUTINE interpolate_fcor
