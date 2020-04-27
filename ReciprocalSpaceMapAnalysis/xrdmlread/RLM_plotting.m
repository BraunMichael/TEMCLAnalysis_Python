%------------------------------------------------------%
% example2.m   An exaple how an XRDML file can be      %
%              loaded using the XRDMLread function.    %
%                                                      %   
% Demonstration of processing a two axes measurement.  %   
%                                                      %
% version 1.1, 19.11.2010                              %
% (c) Zdenek Matej, Milan Dopita                       %
%------------------------------------------------------%
clear all;
clc;
format long;

d = XRDMLread('RLM_01_1.xrdml') %#ok<NOPTS>

figure

% create an appropriate log scale
v = logspace( mean(d.data(d.data<5)) , log10(max(max(d.data))) , 21 );
col = hsv(21);

% calculate the contour matrix
C = contourc( d.Theta2(1,:) , d.Omega(:,1)-d.Theta2(:,1)/2 , ...
              d.data , v );

% transform contours coordinates into the Q-space
% and draw them
H = []; nn = 0; hold on
while nn+1<size(C,2)
    % extract data
    value = C(1,nn+1);
    level = find(v == value);
    dim = C(2,nn+1);
    ind = nn+1+(1:dim);
    x = C(1,ind); y = C(2,ind); 
    % transformation
    Qx = 2*pi/d.Lambda*(-cos((x/2-y)*pi/180) + cos((x/2+y)*pi/180) );
    Qz = 2*pi/d.Lambda*( sin((x/2-y)*pi/180) + sin((x/2+y)*pi/180) );
    % draw line
    H(end+1) = line(Qx,Qz,'Color',col(level,:)); %#ok<SAGROW>
    % draw patch (Matlab)
    %H(end+1) = patch(Qx,Qz,col(level,:),'Line','none'); %#ok<SAGROW>
    % save data
    C(1,ind) = Qx;
    C(2,ind) = Qz;
    out(1,ind)=Qx;
    out(2,ind)=Qz;
    out(3,ind)=value;
    out_real=out.';
    % increase index
    nn = nn+dim+1;
end

axis square
box on
xlabel('Q_x (1/A)')
ylabel('Q_z (1/A)')

title( d.filename ,'Interpreter','none','FontName','Helvetica')
