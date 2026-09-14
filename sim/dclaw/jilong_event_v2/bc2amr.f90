! Case-level finite debris-flow inlet.  This replaces only GeoClaw's boundary
! wrapper; D-Claw constitutive and Riemann source code are unchanged.
subroutine bc2amr(val,aux,nrow,ncol,meqn,naux,hx,hy,level,time,xlo_patch,xhi_patch,ylo_patch,yhi_patch)
 use amr_module, only: mthbc,xlower,ylower,xupper,yupper,xperdom,yperdom,spheredom
 implicit none
 integer,intent(in)::nrow,ncol,meqn,naux,level
 real(kind=8),intent(in)::hx,hy,time,xlo_patch,xhi_patch,ylo_patch,yhi_patch
 real(kind=8),intent(inout)::val(meqn,nrow,ncol),aux(naux,nrow,ncol)
 integer::i,j,nxl,nxr,nyb,nyt,ibeg,jbeg,ios
 real(kind=8)::hxmarg,hymarg,qpeak,duration,u,v,y,entry_y,width,h,m,pb
 logical,save::loaded=.false.
 real(kind=8),save::qp,td
 hxmarg=hx*.01d0; hymarg=hy*.01d0
 if (.not.loaded) then
   open(11,file='inflow.data',status='old',iostat=ios); if (ios.ne.0) stop 'missing inflow.data'
   read(11,*) qp; read(11,*) td; close(11); loaded=.true.
 endif
 if (xperdom .and. (yperdom .or. spheredom)) return
 ! Default all physical edges to zero-order extrapolation, including auxiliaries.
 if (xlo_patch.lt.xlower-hxmarg) then
   nxl=int((xlower+hxmarg-xlo_patch)/hx)
   do j=1,ncol; do i=1,nxl; aux(:,i,j)=aux(:,nxl+1,j); val(:,i,j)=val(:,nxl+1,j); enddo; enddo
   if (mthbc(1).eq.0) then
     entry_y=3143392.5d0; width=256.d0; h=10.d0; m=.63d0; pb=1000.d0*9.81d0*h
     if (time.ge.0.d0 .and. time.le.td) then
       ! triangular Q(t): qpeak at td/2; Q=h*u*width through vertical inlet.
       qpeak=qp*(1.d0-abs(2.d0*time/td-1.d0)); u=qpeak/(h*width); v=-.6472d0/.7623d0*u
       do j=1,ncol
         y=ylo_patch+(j-.5d0)*hy
         if (abs(y-entry_y).le.width/2.d0) then
           do i=1,nxl
             val(:,i,j)=0.d0; val(1,i,j)=h; val(2,i,j)=h*u; val(3,i,j)=h*v
             val(4,i,j)=h*m; val(5,i,j)=pb; val(6,i,j)=h*.5d0; val(7,i,j)=0.d0
           enddo
         endif
       enddo
     endif
   endif
 endif
 if (xhi_patch.gt.xupper+hxmarg) then
   nxr=int((xhi_patch-xupper+hxmarg)/hx); ibeg=max(nrow-nxr+1,1)
   do i=ibeg,nrow; do j=1,ncol; aux(:,i,j)=aux(:,ibeg-1,j); val(:,i,j)=val(:,ibeg-1,j); enddo; enddo
 endif
 if (ylo_patch.lt.ylower-hymarg) then
   nyb=int((ylower+hymarg-ylo_patch)/hy)
   do j=1,nyb; do i=1,nrow; aux(:,i,j)=aux(:,i,nyb+1); val(:,i,j)=val(:,i,nyb+1); enddo; enddo
 endif
 if (yhi_patch.gt.yupper+hymarg) then
   nyt=int((yhi_patch-yupper+hymarg)/hy); jbeg=max(ncol-nyt+1,1)
   do j=jbeg,ncol; do i=1,nrow; aux(:,i,j)=aux(:,i,jbeg-1); val(:,i,j)=val(:,i,jbeg-1); enddo; enddo
 endif
end subroutine bc2amr
