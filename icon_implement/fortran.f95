! Description: Interpolate sub-grid shortwave correction factor for specific
!              sun position (Fortran version; implement in ICON in:
!              icon-nwp/src/atm_phy_schemes/mo_radiation.f90)
!
! Compilation:
! f2py -c --fcompiler=gnu95 fortran.f95 -m fortran
! f2py -c --fcompiler=gnu95 --opt=-O3 fortran.f95 -m fortran
!
! Author: Christian Steger, May 2026

SUBROUTINE interpolate_fcor(horizon, swdir_cor, terrain_normal, &
                            ztheta_sun, zphi_sun, fcor_sun, zha_sun)

  IMPLICIT NONE

  ! 8 byte real kind parameter
  INTEGER, PARAMETER :: dp = selected_real_kind(13)
  ! 4 byte real kind parameter
  INTEGER, PARAMETER :: sp = selected_real_kind(6)
  ! INTEGER, PARAMETER :: wp = sp  ! single precision
  INTEGER, PARAMETER :: wp = dp  ! double precision

  REAL(wp), DIMENSION(72) :: horizon ! (24 * 3; 3 horizon angles (min, mean, max)) [deg]
  REAL(wp), DIMENSION(120) :: swdir_cor ! (24 * 5; correction factors) [-]
  REAL(wp), DIMENSION(3) :: terrain_normal ! x, y and z-component [-]
  REAL(wp) :: ztheta_sun ! sun elevation angle [rad]
  REAL(wp) :: zphi_sun   ! sun azimuth angle [rad]
  REAL(wp) :: zha_sun    ! interpolated horizon angle below the sun [deg]
  REAL(wp) :: fcor_sun   ! interpolated fcor-value for sun position [-] (new variable)

  ! Useful variables already defined in ICON
  INTEGER :: k, ii
  REAL(wp) :: pi, rad2deg
  REAL(wp) :: zihor

  ! New local scalar variables
  REAL(wp) :: eta, horizon_min, horizon_max, pos_norm, fcor_left, fcor_right, &
              angle_left, angle_right, deg2rad, weight_left
  INTEGER :: num_azim, num_nodes, idx_left, idx_right, idx_lin

  ! New local vector variables
  REAL(wp), DIMENSION(3) :: sun_vec
  REAL(wp), DIMENSION(2) :: fcor
  INTEGER, DIMENSION(2) :: azim_idx

  ! Constant values
  num_azim = 24
  eta = 2.0_wp
  num_nodes = 7 ! all interpolation nodes including bounds

  !f2py threadsafe
  !f2py intent(in) horizon
  !f2py intent(in) swdir_cor
  !f2py intent(in) terrain_normal
  !f2py intent(in) ztheta_sun
  !f2py intent(in) zphi_sun
  !f2py intent(out) fcor_sun
  !f2py intent(out) zha_sun

  ! Constants
  pi = 4.0_wp * ATAN(1.0_wp) ! -> where defined in ICON?
  rad2deg = 180.0_wp / pi
  deg2rad = pi / 180.0_wp

  ! Compute relevant azimuth indices
  zihor = REAL(INT(360.0_wp / num_azim), wp) ! 15 deg
  ii = MIN(num_azim - 1, INT(rad2deg * zphi_sun / zihor))
  k = MOD(ii + 1, num_azim)
  azim_idx = (/ ii, k /) ! indices start with 0!

  ! Horizon for binary shadow mask
  weight_left = (zihor * (ii + 1) - rad2deg * zphi_sun) / zihor
  zha_sun = horizon(ii * 3 + 2) * weight_left + horizon(k * 3 + 2) * (1.0 - weight_left)

  ! Compute direct shortwave correction factors for both azimuth directions
  DO k = 1, 2

    horizon_min = horizon(azim_idx(k) * 3 + 1) ! minimal horizon angle [deg]
    horizon_max = horizon(azim_idx(k) * 3 + 3) ! maximal horizon angle [deg]
    pos_norm = (ztheta_sun * rad2deg - horizon_min) / (horizon_max - horizon_min)
    ! horizon_max > horizon_min -> otherwise, division by zero! ###############
    ! -> also relevant for below expression '(angle_right - angle_left)'

    ! Total shadow -> fcor = 0.0
    IF (pos_norm <= 0.0_wp) THEN
      fcor(k) = 0.0_wp

    ! No shadow -> compute fcor from terrain normal
    ELSEIF (pos_norm >= 1.0_wp) THEN
      sun_vec = (/ COS(ztheta_sun) * SIN(deg2rad*azim_idx(k) * zihor), &
                   COS(ztheta_sun) * COS(deg2rad*azim_idx(k) * zihor), &
                   SIN(ztheta_sun) /)
      fcor(k) = (1.0_wp / MAX(sun_vec(3), 1e-5_wp)) * DOT_PRODUCT(sun_vec, terrain_normal)

    ! Partial shadow -> interpolate fcor from saved data
    ELSE
      idx_left = INT((num_nodes - 1) * pos_norm ** (1.0_wp / eta)) + 1
      idx_left = MIN(idx_left, num_nodes - 1) ! not sure if actually needed...
      idx_right = idx_left + 1
      angle_left = (horizon_min + (horizon_max - horizon_min) &
        * (REAL(idx_left - 1, wp) / REAL(num_nodes - 1, wp)) ** eta) ! [deg]
      angle_right = (horizon_min + (horizon_max - horizon_min) &
        * (REAL(idx_right - 1, wp) / REAL(num_nodes - 1, wp)) ** eta) ! [deg]
      IF (idx_left == 1) THEN
        fcor_left = 0.0_wp
        idx_lin = (azim_idx(k) * (num_nodes - 2) + idx_right) - 1
        fcor_right = swdir_cor(idx_lin)
      ELSEIF (idx_right == num_nodes) THEN
        idx_lin = (azim_idx(k) * (num_nodes - 2) + idx_left) - 1
        fcor_left = swdir_cor(idx_lin)
        sun_vec = (/ COS(deg2rad * angle_right) * SIN(deg2rad * azim_idx(k) * zihor), &
                     COS(deg2rad * angle_right) * COS(deg2rad * azim_idx(k) * zihor), &
                     SIN(deg2rad * angle_right) /)
        fcor_right = (1.0_wp / MAX(sun_vec(3), 1e-5_wp)) * DOT_PRODUCT(sun_vec, terrain_normal)
      ELSE
        idx_lin = (azim_idx(k) * (num_nodes - 2) + idx_left) - 1
        fcor_left = swdir_cor(idx_lin)
        idx_lin = (azim_idx(k) * (num_nodes - 2) + idx_right) - 1
        fcor_right = swdir_cor(idx_lin)
      ENDIF
      weight_left = ((angle_right - ztheta_sun * rad2deg) / (angle_right - angle_left))
      fcor(k) = (fcor_left * weight_left + fcor_right * (1.0_wp - weight_left))
    ENDIF
  END DO

  ! Interpolate fcor-value at azimuth angle
  weight_left = (zihor * (ii + 1) - rad2deg * zphi_sun) / zihor
  fcor_sun = fcor(1) * weight_left + fcor(2) * (1.0 - weight_left)

END SUBROUTINE interpolate_fcor
