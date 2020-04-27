clear all; close all; format shortg; 
set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultAxesFontSize',24)
resolution = '-r300'; %Resolution of png output files


LatticeConstant_Ge = 5.65791; %in Å need to check with notebook
LatticeConstant_Sn = 6.4892; %in Å need to confirm
h = 3; k = 3; l = 3; %Peak of interest
Cu_kalpha1 = 1.5405929; %in Å, Holzer et al PRA 56,6 (1997) https://journals.aps.org/pra/abstract/10.1103/PhysRevA.56.4554

cd('C:\Michael\Stanford\Research\Data\XRD')
xyFilenames = {}; data = {};
firstAngles = []; combinedData = []; outputData = [];
ii = 1;
[xyFilenames{ii}, xyFolderpath] = uigetfile('*.xy', 'Choose the first xy file');
scalingFactor = str2double(cell2mat(inputdlg({sprintf('Enter scaling value (for different counting times) of %s. Multiplies the intensity of the file just selected by this value',xyFilenames{ii})},'Scaling Factor',[1 70],{'1'})));

cd(xyFolderpath)


extractSampleNameString = cell2mat(xyFilenames(ii));
nameSearchString = 'MB';
nameStartIndex = strfind(extractSampleNameString,nameSearchString);
if isempty(nameStartIndex)
    nameSearchString = 'JZL';
    nameStartIndex = strfind(extractSampleNameString,nameSearchString);
end

if isempty(nameStartIndex)
    sampleName = (cell2mat(inputdlg({'Enter sample name'},'Enter sample name',[1 70],{''})));
else
    remainingSampleString = extractSampleNameString(nameStartIndex+length(nameSearchString):end);
    possibleSampleNumbers = regexp(remainingSampleString,'\d*','Match');
    sampleNumber = cell2mat(possibleSampleNumbers(1));
    sampleName = strcat(nameSearchString, sampleNumber);
end
sampleName = strcat(sampleName, '_CombinedXRD');

data{ii} = load(xyFilenames{ii});
data{ii}(:,2) = scalingFactor .* data{ii}(:,2);
firstAngles(ii) = data{ii}(1,1);

button = 'yes';
while strcmp(button, 'yes')
    button=questdlg('Do you have more .xy files for this scan?','Select additional .xy files?','yes','no','no');
    switch button
        case 'yes'
            ii = ii + 1;
            [xyFilenames{ii}, ~] = uigetfile('*.xy', 'Choose the next xy file');
            scalingFactor = str2double(cell2mat(inputdlg({sprintf('Enter scaling value (for different counting times) of %s. Multiplies the intensity of the file just selected by this value',xyFilenames{ii})},'Scaling Factor',[1 70],{'1'})));
            data{ii} = load(xyFilenames{ii});
            data{ii}(:,2) = scalingFactor .* data{ii}(:,2);
            firstAngles(ii) = data{ii}(1,1);
        case 'no'
    end
end

[~,fileIndicies] = sort(firstAngles);

for jj = 1:length(fileIndicies)
   outputData = [outputData; data{fileIndicies(jj)}];
end


d_hkl_Ge = sqrt((LatticeConstant_Ge^2)/(h^2+k^2+l^2));
substrateReferencePosition_2Theta = 2*asind(Cu_kalpha1/(2*d_hkl_Ge));


substrateOffsetFitFigure = figure('Name','SubstrateOffsetFitting');
semilogy(outputData(:,1),outputData(:,2),'k')
questdlg('Draw a rectangle around the Ge(333) substrate peak. Only the x-axis locations matter.','Select substrate peak','ok','ok');
selectionRectangle = getrect;
lowAngle = selectionRectangle(1);
highAngle = lowAngle + selectionRectangle(3);
peakData = outputData(outputData(:,1) > lowAngle & outputData(:,1) < highAngle,:);
substrateFit = fit(peakData(:,1), peakData(:,2), 'gauss1');
substratePeakPosition = substrateFit.b1;
peakOffset = substratePeakPosition - substrateReferencePosition_2Theta;
close SubstrateOffsetFitting

figure ()
pause(0.00001);
frame_h = get(handle(gcf),'JavaFrame');
set(frame_h,'Maximized',1);
outputData(:,1) = outputData(:,1) + peakOffset;
semilogy(outputData(:,1),outputData(:,2),'k'); hold on

tt = 1;
button = 'yes';
while strcmp(button, 'yes')
    button=questdlg('Choose another GeSn peak? Draw a rectangle around the film peak of interest. Only the x-axis locations matter.','Select GeSn peaks','yes','no','no');
    switch button
        case 'yes'
            selectionRectangle = getrect;
            lowAngle = selectionRectangle(1);
            highAngle = lowAngle + selectionRectangle(3);
            peakData = outputData(outputData(:,1) > lowAngle & outputData(:,1) < highAngle,:);
            
            sampleFit = fit(peakData(:,1), peakData(:,2), 'gauss1');
            samplePeakPosition_2Theta = sampleFit.b1;
            sampleLattice_d = Cu_kalpha1/(2*sind(0.5*samplePeakPosition_2Theta));
            sampleLatticeParameter = sqrt((sampleLattice_d^2)*(h^2+k^2+l^2));
            sampleSnContent = interp1([LatticeConstant_Ge LatticeConstant_Sn], [0 100], sampleLatticeParameter); %Vegard's law for Sn content
            
            fitCurve = feval(sampleFit, peakData(:,1));
            semilogy(peakData(:,1), fitCurve, 'r')
            text(samplePeakPosition_2Theta,sampleFit.a1,strcat('\leftarrow', num2str(sampleSnContent),'%'),'Rotation',90,'Interpreter','tex')
            tt = tt + 1;
        case 'no'
    end
end

hold off


if exist(strcat(pwd, '\', sampleName,'.png'), 'file') == 2
    delete(strcat(sampleName,'.png'))
end
print(strcat(sampleName,'.png'),'-dpng',resolution)


normalizedOutputData = outputData(:,2)/max(outputData(:,2));
outputFile = fopen(strcat(sampleName,'.txt'),'wt');
fprintf(outputFile, '2Theta (Degrees)\tIntensity (Counts)\tNormalized Intensity (arb units)\n');
for nn = 1:length(outputData(:,1))
    fprintf(outputFile, '%f\t%.0f\t%g\n', outputData(nn,1), outputData(nn,2), normalizedOutputData(nn));
end
fclose(outputFile);