%Michael Braun
%Takes in XRDML files from X'Pert and makes transforms the data to
%something usable
%Uses XRDMLread from Zdenek Matej, Milan Dopita http://www.xray.cz/xrdmlread/
clear all;
clc;
format short;

d = XRDMLread('RLM_01_1.xrdml') %#ok<NOPTS>

% create an appropriate log scale
twotheta=d.Theta2;
omega=d.Omega;
intensity=d.data+1; %+1 to remove white, everything is now defined when taking log10
Qx_simple=2*pi/d.Lambda*(-cos((twotheta-omega)*pi/180) + cos((omega)*pi/180) ); %Remove _simple after definitely getting rid of commented code at bottom
Qz_simple=2*pi/d.Lambda*( sin((twotheta-omega)*pi/180) + sin((omega)*pi/180) );

figure(1)
axis square
box on
surf(Qx_simple,Qz_simple,log10(intensity),'Linestyle','none','FaceColor','interp')
view(0,90);
xlabel('$$Q_{x}\ (1/\textrm{\AA}$$)','interpreter','LaTeX','fontsize',16)
ylabel('$$Q_{z}\ (1/\textrm{\AA}$$)','interpreter','LaTeX','fontsize',16)
title('Reciprocal Space Map','interpreter','LaTeX','fontsize',16)
c=colorbar;
c.Label.String = 'log_{10}Intensity';
set(c,'fontsize',16)

% 
% % calculate the contour matrix
% C = contourc( d.Theta2(1,:) , d.Omega(:,1)-d.Theta2(:,1)/2 , ...
%               d.data,v);
% % transform contours coordinates into the Q-space
% nn = 0;
% size(C,2)
% while nn+1<size(C,2)
%     % extract data
%     value = C(1,nn+1);
% %     level = find(v == value);
%     dim = C(2,nn+1);
%     ind = nn+1+(1:dim);
%     x = C(1,ind); y = C(2,ind); 
%     % transformation
%     Qx = 2*pi/d.Lambda*(-cos((x/2-y)*pi/180) + cos((x/2+y)*pi/180) );
%     Qz = 2*pi/d.Lambda*( sin((x/2-y)*pi/180) + sin((x/2+y)*pi/180) );
%     % save data
%     C(1,ind) = Qx;
%     C(2,ind) = Qz;
%     vQx(ind)=Qx;
%     vQz(ind)=Qz;
%     vI(ind)=value;
%     % increase index
%     nn = nn+dim+1;
% end
% vQx=vQx';
% vQz=vQz';
% vI=vI';
% out=[vQx vQz vI];
