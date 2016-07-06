!******************************************************************************
!*                     Code generated with sympy 0.7.6.1                      *
!*                                                                            *
!*              See http://www.sympy.org/ for more information.               *
!*                                                                            *
!*                      This file is part of 'autowrap'                       *
!******************************************************************************

subroutine autofunc(a, b, m, x)
implicit none
REAL*8, intent(in) :: a
REAL*8, intent(in) :: b
INTEGER*4, intent(in) :: m
REAL*8, intent(out), dimension(1:m) :: x
INTEGER*4 :: i

do i = 1, m
   x(i) = a + (a - b)*(i - 1)/(-m + 1)
end do

end subroutine
