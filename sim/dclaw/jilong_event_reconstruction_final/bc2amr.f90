! Case-local north-boundary unit-Froude inflow; no global D-Claw source modified.
subroutine bc2amr(val,aux,nrow,ncol,meqn,naux,hx,hy,level,time,xlo_patch,xhi_patch,ylo_patch,yhi_patch)
 use amr_module, only: mthbc,xlower,ylower,xupper,yupper,xperdom,yperdom,spheredom
 implicit none
 integer,intent(in)::nrow,ncol,meqn,naux,level
 real(kind=8),intent(in)::hx,hy,time,xlo_patch,xhi_patch,ylo_patch,yhi_patch
 real(kind=8),intent(inout)::val(meqn,nrow,ncol),aux(naux,nrow,ncol)
 integer::i,j,nxl,nxr,nyb,nyt,ibeg,jbeg,ios
 real(kind=8)::hxmarg,hymarg,qp,td,tx,ty,xmin,xmax,m0,rhof,grav,dryt,frn,q,h,un,utotal,u,v,x,pb,bwidth
 logical,save::loaded=.false.
 real(kind=8),save::sqp,std,stx,sty,sxmin,sxmax,sm0,srhof,sgrav,sdryt,sfrn
 hxmarg=hx*.01d0; hymarg=hy*.01d0
 if (.not.loaded) then
   open(11,file='inflow.data',status='old',iostat=ios); if (ios.ne.0) stop 'missing inflow.data'
   read(11,*) sqp; read(11,*) std; read(11,*) stx; read(11,*) sty
   read(11,*) sxmin; read(11,*) sxmax; read(11,*) sm0; read(11,*) srhof
   read(11,*) sgrav; read(11,*) sdryt; read(11,*) sfrn; close(11); loaded=.true.
 endif
 if (xperdom .and. (yperdom .or. spheredom)) return
 if (xlo_patch.lt.xlower-hxmarg) then
   nxl=int((xlower+hxmarg-xlo_patch)/hx)
   do j=1,ncol; do i=1,nxl; aux(:,i,j)=aux(:,nxl+1,j); val(:,i,j)=val(:,nxl+1,j); enddo; enddo
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
   do j=jbeg,ncol
     do i=1,nrow
       aux(:,i,j)=aux(:,i,jbeg-1); val(:,i,j)=val(:,i,jbeg-1)
       val(3,i,j)=-val(3,i,jbeg-1)
     enddo
   enddo
   if (mthbc(4).eq.0 .and. time.gt.0.d0 .and. time.lt.std) then
     qp=sqp*dsin(acos(-1.d0)*time/std)
     bwidth=sxmax-sxmin+hx
     if (qp.gt.bwidth*sfrn*dsqrt(sgrav)*sdryt**1.5d0) then
       h=(qp/(bwidth*sfrn*dsqrt(sgrav)))**(2.d0/3.d0)
       un=sfrn*dsqrt(sgrav*h); utotal=un/(-sty); u=utotal*stx; v=utotal*sty; pb=srhof*sgrav*h
       do i=1,nrow
         x=xlo_patch+(i-.5d0)*hx
         if (x.ge.sxmin-.25d0*hx .and. x.le.sxmax+.25d0*hx) then
           do j=jbeg,ncol
             val(:,i,j)=0.d0
             val(1,i,j)=h; val(2,i,j)=h*u; val(3,i,j)=h*v; val(4,i,j)=sm0*h
             val(5,i,j)=pb; val(6,i,j)=0.d0; val(7,i,j)=0.d0
           enddo
         endif
       enddo
     endif
   endif
 endif
end subroutine bc2amr
