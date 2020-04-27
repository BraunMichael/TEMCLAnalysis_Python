clear all;clc;

sample_base = input ('Enter sample ID : ','s');
omega_start = input ('Enter starting omega : ');
omega_end = input ('Enter ending omega : ');
file_step = input ('Enter step size : ');
range = omega_end - omega_start;
offset = -0.5*range;
qx=[];qz=[];intensity=[];two_theta_mat=[];omega_mat=[];
count=0;

disp(' ');disp(' ');disp('Processing file(s) : ');disp(' ');

r1=1/3.905;r2=r1*1.5406/2;

for file_no = omega_start:file_step:omega_end

    if length(num2str(file_no))==5
        scan_file = strcat(sample_base,'_RSM_011 Omega=',num2str(file_no),'.csv');
    elseif length(num2str(file_no))==4
        scan_file = strcat(sample_base,'_RSM_011 Omega=',num2str(file_no),'0','.csv');
    elseif length(num2str(file_no))==2
        scan_file = strcat(sample_base,'_RSM_011 Omega=',num2str(file_no),'.00','.csv');
    end

    fin = fopen(strcat(sample_base,'_RSM_011_scan files\',scan_file),'r');
    for i=1:29,  buffer = fgetl(fin);  end

    data=[];
    while(1)
        next=fgetl(fin);
        if next==-1
            break;
        else
            data=[data;str2num(next)];
        end
    end

    two_theta = data(:,1);
    omega = 0.5*two_theta + offset;
    
    two_theta_mat = [two_theta_mat two_theta];
    omega_mat = [omega_mat omega];

    qx=[qx 1E4*r2*(cos((pi/180)*omega)-cos((pi/180)*(two_theta-omega)))];
    qz=[qz 1E4*r2*(sin((pi/180)*omega)+sin((pi/180)*(two_theta-omega)))];
    intensity=[intensity data(:,2)];

    offset = offset+file_step;
    fclose(fin);
    count=count+1;
    disp(scan_file);
end

disp(' ');disp(' ');disp(strcat('Number of files processed : ',num2str(count)));disp(' ');

surf (two_theta_mat,omega_mat,log10(intensity));
view(0,90);
xlabel('2Theta/Omega');ylabel('Omega');
axis('tight');
figure;

surf (qx,qz,log10(intensity));
view(0,90);
xlabel('Qx*10000');ylabel('Qz*10000');
axis('tight');