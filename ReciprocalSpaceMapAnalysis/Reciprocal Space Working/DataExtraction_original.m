%Michael Braun
%Takes in XRDML files from X'Pert and makes transforms the data to
%something usable
%Uses XRDMLread from Zdenek Matej, Milan Dopita http://www.xray.cz/xrdmlread/
clear all;
clc;
format long;

d = XRDMLread('RLM_01_1.xrdml') %#ok<NOPTS>

% create an appropriate log scale
v = logspace( mean(d.data(d.data<5)) , log10(max(max(d.data))) , 21 );
col = hsv(21);

% calculate the contour matrix
C = contourc( d.Theta2(1,:) , d.Omega(:,1)-d.Theta2(:,1)/2 , ...
              d.data,v);

% transform contours coordinates into the Q-space
nn = 0;
size(C,2)
while nn+1<size(C,2)
    % extract data
    value = C(1,nn+1);
%     level = find(v == value);
    dim = C(2,nn+1);
    ind = nn+1+(1:dim);
    x = C(1,ind); y = C(2,ind); 
    % transformation
    Qx = 2*pi/d.Lambda*(-cos((x/2-y)*pi/180) + cos((x/2+y)*pi/180) );
    Qy = 2*pi/d.Lambda*( sin((x/2-y)*pi/180) + sin((x/2+y)*pi/180) );
    % save data
    C(1,ind) = Qx;
    C(2,ind) = Qy;
    vQx(ind)=Qx;
    vQy(ind)=Qy;
    vI(ind)=value;
    % increase index
    nn = nn+dim+1;
end

% qxi=linspace(min(vQx),max(vQx),3000);
% qyi=linspace(min(vQy),max(vQy),3000);
% [qXI,qYI]=meshgrid(qxi,qyi);
% ZI = griddata(vQx,vQy,vI,qXI,qYI);

% 
% figure
% % contourf(qXI,qYI,qZI)
% axis square
% box on
% surf(Qx,Qy,ZI)
% xlabel('Q_x (1/A)')
% ylabel('Q_y (1/A)')
% title( d.filename ,'Interpreter','LaTeX')
