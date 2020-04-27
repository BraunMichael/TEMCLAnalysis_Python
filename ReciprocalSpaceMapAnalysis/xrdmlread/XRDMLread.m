function [data,tree] = XRDMLread(filename)
%--------------------------------------------------------
% XRDMLREAD    Read an 'xrdml' file.
%
% [data,tree] = XRDMLread(filename)
%
% filename - input filename (as string)
%            or directly an XMLTreee Object
%
% data     - data struct (intensities normalised to cps)
% tree     - XMLTree Object
%
% See also XMLTREE
%
% version 1.3.1, 3.2.2011, http://www.xray.cz/xrdmlread/
% Authors: Zdenek Matej, Milan Dopita
% Distributed under the Simplified BSD License.
%--------------------------------------------------------

% Copyright (c) 2011, Zdenek Matej, Milan Dopita
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions
% are met:
%
%   1. Redistributions of source code must retain the above copyright
%   notice, this list of conditions and the following disclaimer.
%   2. Redistributions in binary form must reproduce the above copyright
%   notice, this list of conditions and the following disclaimer in the
%   documentation and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
% TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
% A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
% HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
% TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
% THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% check parameters and load the xmltree object
if ~exist('filename','var'), error('XRDMLread:general','no input specified'), end

if ischar(filename),
    % input is a filename - create the xmltree object
    if ~exist(filename,'file'),
        error('XRDMLread:general','input file does not exist'),
    end
    tree = xmltree(filename);
elseif strcmpi( class(filename) , 'xmltree' ),
    % XMLTree Object already loaded, use the supplied object
    tree = filename;
    filename = getfilename(tree);  %#ok<NASGU>
else
    % unknow option
    error('XRDMLread:general', ...
          'input argument is neither string nor xmltree object'),
end

% get filename
data.filename = getfilename(tree);
 
% get sample ID
xpath = '/xrdMeasurements/sample/id';
uid = find(tree,xpath);
data.sampleID = getNodeVal(tree, uid);

% get comment
xpath = '/xrdMeasurements/comment';
uid = find(tree,xpath);
uid_child = children(tree,uid);
data.comment = '';
for uid=uid_child
    node = get(tree,uid);
    if strcmp(node.name,'entry')
        data.comment = strvcat(data.comment, getNodeVal(tree, uid)); %#ok<REMFF1>
    end
end
%if ~isempty(data.comment) data.comment = data.comment(1:end-2); end

% get nb. of scans
xpath = '/xrdMeasurements/xrdMeasurement/scan';
uid_scans = find(tree,xpath);
nb_scans = length(uid_scans);

% get (h k l) [OPTIONAL]
if (nb_scans>0)
    xpath = '/xrdMeasurements/xrdMeasurement/scan[1]/reflection';
    uid = find(tree,xpath);
    if ~isempty(uid)
        uid = find(tree,[xpath '/hkl/h']);
        data.hkl(1) = str2double(getNodeVal(tree, uid));
        uid = find(tree,[xpath '/hkl/k']);
        data.hkl(2) = str2double(getNodeVal(tree, uid));
        uid = find(tree,[xpath '/hkl/l']);
        data.hkl(3) = str2double(getNodeVal(tree, uid));
    else
        % data.hkl = [0 0 0]; % optional
    end
end

% get measurement type
xpath = '/xrdMeasurements/xrdMeasurement';
uid = find(tree,xpath);
data.measType = getAttribVal(tree, uid, 'measurementType');

% if not a simple scan and not the 'Repeated scan' than get
% the step axis type
if ~strcmp(data.measType,'Scan') && ~strcmp(data.measType,'Repeated scan'),
    xpath = '/xrdMeasurements/xrdMeasurement';
    uid = find(tree,xpath);
    data.stepAxis = getAttribVal(tree, uid, 'measurementStepAxis');
end

% get the scan axis type
if (nb_scans>0)
    data.scanAxis = getAttribVal(tree, uid_scans(1), 'scanAxis');
end

% get scans
data.scannb = []; data.data = []; data.time = []; data.Theta2 = []; data.Omega = [];
data.Phi = []; data.Psi = []; data.X = []; data.Y = []; data.Z = [];
data.iscannb = {}; data.idata = {}; data.itime = {}; data.iTheta2 = {}; data.iOmega = {};
data.iPhi = {}; data.iPsi = {}; data.iX = {}; data.iY = {}; data.iZ = {};
for k=1:nb_scans
    scan = getScan(tree,k);
    if ~isempty(scan)
        if strcmp(data.measType,'Scan') || strcmp(scan.status,'Completed') % add to scans
            data.scannb(end+1,:) = k;
            data.data(end+1,:) = scan.data;
            data.time(end+1,:) = scan.time;
            data.Theta2(end+1,:) = scan.Theta2;
            data.Omega(end+1,:) = scan.Omega;
            if isfield(scan,'Phi'), data.Phi(end+1,:) = scan.Phi; end
            if isfield(scan,'Psi'), data.Psi(end+1,:) = scan.Psi; end
            if isfield(scan,'X'), data.X(end+1,:) = scan.X; end
            if isfield(scan,'Y'), data.Y(end+1,:) = scan.Y; end
            if isfield(scan,'Z'), data.Z(end+1,:) = scan.Z; end
        else % add to incomplete scans
            data.iscannb{end+1} = k;
            data.idata{end+1} = scan.data;
            data.itime{end+1} = scan.time;
            data.iTheta2{end+1} = scan.Theta2;
            data.iOmega{end+1} = scan.Omega;
            if isfield(scan,'Phi'), data.iPhi{end+1} = scan.Phi; end
            if isfield(scan,'Psi'), data.iPsi{end+1} = scan.Psi; end
            if isfield(scan,'X'), data.iX{end+1} = scan.X; end
            if isfield(scan,'Y'), data.iY{end+1} = scan.Y; end
            if isfield(scan,'Z'), data.iZ{end+1} = scan.Z; end
        end
    end
end
% if we have only one incoplete scan, the scan is considered to be
% completed and is moved to completed scans list
if isempty(data.scannb) && length(data.iscannb)==1,
    warning('XRDMLread:incompleteScan','one and only incoplete scan found in the data, the scan considered as completed')
    data.scannb = data.iscannb; data.iscannb = [];
    data.data = data.idata{1}; data.idata = {};
    data.time = data.itime{1}; data.itime = {};
    data.Theta2 = data.iTheta2{1}; data.iTheta2 = {};
    data.Omega = data.iOmega{1}; data.iOmega = {};
    if isfield(data,'iPhi') && ~isempty(data.iPhi), data.Phi = data.iPhi{1}; data.iPhi = {}; end
    if isfield(data,'iPsi') && ~isempty(data.iPsi), data.Psi = data.iPsi{1}; data.iPsi = {}; end
    if isfield(data,'iX') && ~isempty(data.iX), data.X = data.iX{1}; data.iX = {}; end
    if isfield(data,'iY') && ~isempty(data.iY), data.Y = data.iY{1}; data.iY = {}; end
    if isfield(data,'iZ') && ~isempty(data.iZ), data.Z = data.iZ{1}; data.iZ = {}; end
end

% remove redundant information
if isempty(data.Phi), data = rmfield(data,'Phi'); end
if isempty(data.Psi), data = rmfield(data,'Psi'); end
if isempty(data.X), data = rmfield(data,'X'); end
if isempty(data.Y), data = rmfield(data,'Y'); end
if isempty(data.Z), data = rmfield(data,'Z'); end
if isempty(data.iscannb)
    data = rmfield(data,'iscannb'); data = rmfield(data,'idata'); data = rmfield(data,'itime'); 
    data = rmfield(data,'iTheta2 '); data = rmfield(data,'iOmega'); data = rmfield(data,'iPhi'); 
    data = rmfield(data,'iPsi'); data = rmfield(data,'iX'); data = rmfield(data,'iY'); data = rmfield(data,'iZ');
end
if all(data.time==data.time(1)), data.time = data.time(1); end
if all(data.Theta2==data.Theta2(1)), data.Theta2 = data.Theta2(1); end
if all(data.Omega==data.Omega(1)), data.Omega = data.Omega(1); end
if (isfield(data,'Phi') && all(data.Phi==data.Phi(1))), data.Phi = data.Phi(1); end
if (isfield(data,'Psi') && all(data.Psi==data.Psi(1))), data.Psi = data.Psi(1); end
if (isfield(data,'X') && all(data.X==data.X(1))), data.X = data.X(1); end
if (isfield(data,'Y') && all(data.Y==data.Y(1))), data.Y = data.Y(1); end
if (isfield(data,'Z') && all(data.Z==data.Z(1))), data.Z = data.Z(1); end

% in case of 'Repeated scan' sum all completed scans together and remove redundant data
if strcmp(data.measType,'Repeated scan')
    % average completed scans (intensity is in cps)
    for k=2:length(data.scannb)
        data.data(1,:) = data.data(1,:) + data.data(k,:);
    end
    data.data = data.data(1,:)/length(data.scannb);
    % reduce all possible axis (TODO:: check if they are really equal)
    if (isfield(data,'Theta2')), data.Theta2 = data.Theta2(1,:); end
    if (isfield(data,'Omega')), data.Omega = data.Omega(1,:); end
    if (isfield(data,'Phi')), data.Phi = data.Phi(1,:); end
    if (isfield(data,'Psi')), data.Psi = data.Psi(1,:); end
    if (isfield(data,'X')), data.X = data.X(1,:); end
    if (isfield(data,'Y')), data.Y = data.Y(1,:); end
    if (isfield(data,'Z')), data.Z = data.Z(1,:); end
    % set true time
    data.time = data.time*length(data.scannb);
    % remove redundant information about scans numbers
    data = rmfield(data,'scannb');
end

% get wavelength
xpath = '/xrdMeasurements/xrdMeasurement/usedWavelength';
uid = find(tree,xpath);
data.kType = getAttribVal(tree, uid, 'intended');
uid = find(tree,[xpath '/kAlpha1']);
data.kAlpha1 = str2double(getNodeVal(tree, uid));
uid = find(tree,[xpath '/kAlpha2']);
data.kAlpha2 = str2double(getNodeVal(tree, uid));
uid = find(tree,[xpath '/ratioKAlpha2KAlpha1']);
data.kAlphaRatio = str2double(getNodeVal(tree, uid));
switch data.kType
case {'K-Alpha 1'}
    data.Lambda = data.kAlpha1;
case {'K-Alpha'}
    data.Lambda = data.kAlpha1+data.kAlphaRatio*data.kAlpha2;
    data.Lambda = data.Lambda/1.5;
otherwise
    warning('XRDMLread:badValue','usedWavelength type not supported (using K-Alpha 1)')
    data.Lambda = data.kAlpha1;
end

% get some usefull information (x/y-label, x/y-units)
if (nb_scans>0)
    if isfield(data,'scanAxis')
        switch data.scanAxis
        case {'Gonio'}
            axisType = '2Theta';
            data.xlabel = '2Theta-Theta';
            if (strcmp(data.measType,'Scan') || ...
                strcmp(data.measType,'Repeated scan')), data.x = data.Theta2; end
        case {'2Theta','2Theta-Omega'}
            axisType = '2Theta';
            data.xlabel = data.scanAxis;
            if (strcmp(data.measType,'Scan') || ...
                strcmp(data.measType,'Repeated scan')), data.x = data.Theta2; end
        case {'Omega','Omega-2Theta'}
            axisType = 'Omega';
            data.xlabel = data.scanAxis;
            if (strcmp(data.measType,'Scan') || ...
                strcmp(data.measType,'Repeated scan')), data.x = data.Omega; end
        case {'Reciprocal Space'}
            axisType = 'Omega';
            data.xlabel = 'Omega';
            if (strcmp(data.measType,'Scan') || ...
                strcmp(data.measType,'Repeated scan')), data.x = data.Omega; end
        case {'Phi','Psi','X','Y','Z'}
            axisType = data.scanAxis;
            data.xlabel = data.scanAxis;
            if strcmp(data.measType,'Scan')
                switch data.scanAxis
                case {'Phi'}
                    data.x = data.Phi;
                case {'Psi'}
                    data.x = data.Psi;
                case {'X'}
                    data.x = data.X;
                case {'Y'}
                    data.x = data.Y;   
                case {'Z'}
                    data.x = data.Z;
                end
            end
        otherwise
            warning('XRDMLread:badValue','scanAxis type not supportrd'),
            axisType = 'unknown';
            data.xlabel = 'unknown';
        end
        xpath = '/xrdMeasurements/xrdMeasurement/scan[1]/dataPoints/positions';
        uid = find(tree,xpath);
        for k=uid
            if strcmp(getAttribVal(tree,k,'axis'),axisType)
                data.xunit = getAttribVal(tree,k,'unit');
                break,
            end
        end
        if ~isfield(data,'xunit'), warning('XRDMLread:missingData','xunit value not assigned'), data.xunit='?'; end
    end
    if isfield(data,'stepAxis')
        switch data.stepAxis
        case {'Gonio'}
            axisType = '2Theta';
            data.ylabel = '2Theta-Theta';
        case {'2Theta','2Theta-Omega'}
            axisType = '2Theta';
            data.ylabel = data.stepAxis;
        case {'Omega','Omega-2Theta'}
            axisType = 'Omega';
            data.ylabel = data.stepAxis;
        case {'Phi','Psi','X','Y','Z'}
            axisType = data.stepAxis;
            data.ylabel = data.stepAxis;
        otherwise
            warning('XRDMLread:badValue','scanAxis type not supported'),
            axisType = 'unknown';
            data.ylabel = 'unknown';
        end
        for k=uid
            if strcmp(getAttribVal(tree,k,'axis'),axisType)
                data.yunit = getAttribVal(tree,k,'unit');
                break,
            end
        end
        xpath = '/xrdMeasurements/xrdMeasurement/scan[1]/dataPoints/positions';
        uid = find(tree,xpath);
        for k=uid
            if strcmp(getAttribVal(tree,k,'axis'),axisType)
                data.yunit = getAttribVal(tree,k,'unit');
                break,
            end
        end
        if ~isfield(data,'yunit'), warning('XRDMLread:missingData','yunit value not assigned'), data.yunit='?'; end
    end
end

% additional usefull information

% Mask Width [OPTIONAL]
xpath = '/xrdMeasurements/xrdMeasurement/incidentBeamPath/mask/width';
uid = find(tree,xpath);
if ~isempty(uid)
    unit = getAttribVal(tree, uid, 'unit');
    if ~strcmpi(unit,'mm'),
        warning('XRDMLread:badValue','Mask Width units are not ''mm''.'),
    end
    data.maskWidth = str2double(getNodeVal(tree, uid));
end

% Divergence slit Height [OPTIONAL]
xpath = '/xrdMeasurements/xrdMeasurement/incidentBeamPath/divergenceSlit/height';
uid = find(tree,xpath);
if ~isempty(uid)
    unit = getAttribVal(tree, uid, 'unit');
    if ~strcmpi(unit,'mm'),
        warning('XRDMLread:badValue','Divergence slit Height units are not ''mm''.'),
    end
    data.slitHeight = str2double(getNodeVal(tree, uid));
end

%
end % XRMLread

function [data] = getScan(tree, scannb)
%------------------------------------------------------
% READSCAN    Extract scan data from the XMLTree class.
%
% [data] = getScan(tree, scannb)
%
% tree   - XMLTree Object
% scannb - number of the extracted scan (implicit=1)
%
% data   - data struct (intensities normalized to cps)
%
% Authors: Zdenek Matej, Milan Dopita
%------------------------------------------------------

% check parameters
if ~isa(tree,'xmltree'), error('The first input argument is not a xmltree object'); end
if (~exist('scannb','var') || isempty(scannb) || scannb < 1), scannb = 1; end

% find the scan in the tree
xpath_scan = ['/xrdMeasurements/xrdMeasurement/scan[' num2str(scannb) ']'];
uid_scan = find(tree,xpath_scan);
if isempty(uid_scan), warning('XRDMLread:missingData','Can not find a specified scan.'); data = []; return, end

% get the scan status
data.status = getAttribVal(tree, uid_scan, 'status');

% get the scan axis type
data.scanAxis = getAttribVal(tree, uid_scan, 'scanAxis');

% get intensities
uid = find(tree,[xpath_scan '/dataPoints/intensities']);
units_intensities = getAttribVal(tree,uid,'unit');
data.data = str2num(getNodeVal(tree,uid)); %#ok<ST2NM>

% get counting time
scan_mode = getAttribVal(tree,uid_scan,'mode');
if strcmp(scan_mode,'Pre-set counts')
    uid = find(tree,[xpath_scan '/dataPoints/countingTimes']);
else
    uid = find(tree,[xpath_scan '/dataPoints/commonCountingTime']);
end
data.time = str2num(getNodeVal(tree,uid)); %#ok<ST2NM>

% normalize intensity units to cps
if strcmp(units_intensities,'counts')
    data.data = data.data./data.time;
end

% get the positions of all axes
xpath = [xpath_scan '/dataPoints/positions'];
uid = find(tree,xpath);
n = length(data.data); % nb. of data points
for k=uid
    %node = get(tree,k);
    info = readAxisInfo(tree,k,n);
    switch info.axis
    case {'2Theta'}
        data.Theta2 = info.data;
    case {'Omega'}
        data.Omega  = info.data;
    case {'Phi'}
        data.Phi    = info.data;
    case {'Psi'}
        data.Psi    = info.data;
    case {'X'}
        data.X      = info.data;
    case {'Y'}
        data.Y      = info.data;
    case {'Z'}
        data.Z      = info.data;
    otherwise
        warning('XRDMLread:badValue','axis type not supported')
    end
end

% 
end % getScan

function [val] = getAttribVal(tree, uid, key)
%------------------------------------------------------
% GETATTRIBVAL    Get attribute value.
%
% [val] = getAttribVal(tree, uid, key)
%
% tree - XMLTree Object
% uid  - node uid
% key  - attribute name
%
% val  - attribute value (as a string), empty
%        if an error occures
%
% Authors: Zdenek Matej, Milan Dopita
%------------------------------------------------------

% check parameters
if ~isa(tree,'xmltree'), error('XRDMLread:badArgument','The first input argument is not an xmltree object'); end
if (~exist('key','var') || isempty(key)), error('XRDMLread:badArgument','the third input argument have to be a valid attribute name.'); end

% find the attribute
attr = attributes(tree,'get',uid);
p = [];
if length(attr)>1
    for k=1:length(attr)
        if strcmp(attr{k}.key,key), p = attr{k}; break, end
    end
else
    if strcmp(attr.key,key), p = attr; end
end
if isempty(p), warning('XRDML:missingData','Can not find the specified attribute'); val = ''; return; end

% get attribute value
val = p.val;
%
end % getAttribVal

function [val] = getNodeVal(tree, uid)
%------------------------------------------------------
% GETNODEVAL    Get node value.
%
% [val] = getNodeVal(tree, uid)
%
% tree - XMLTree Object
% uid  - node uid
%
% val  - node value(s) (as a string), empty if an error
%        occures
%
% Authors: Zdenek Matej, Milan Dopita
%------------------------------------------------------

% check parameters
if ~isa(tree,'xmltree'), error('XRDMLread:badArgument','The first input argument is not an xmltree object'); end

% get node value
uid_child = children(tree,uid);
if isempty(uid_child), val = ''; return, end
node = get(tree,uid_child);
val = node.value;
%
end % getNodeVal

%------------------------------------------------------
% auxiliary functions

function [info] = readAxisInfo(tree,uid,n)
info.axis = getAttribVal(tree, uid, 'axis');
info.unit = getAttribVal(tree, uid, 'unit');
unspaced = true;
uid = children(tree,uid);
for k=uid
    node = get(tree,k);
    switch node.name
    case {'startPosition'}
        info.data(2) = str2double(getNodeVal(tree,k));
    case {'endPosition'}
        info.data(3) = str2double(getNodeVal(tree,k));
    case {'commonPosition'}
        info.data(1) = str2double(getNodeVal(tree,k));
    case {'listPositions'}
        info.data    = str2num(getNodeVal(tree,k)); %#ok<ST2NM>
        unspaced = false;
    otherwise
        warning('XRDMLread:badValue','unsupported tag')
        info.data = [];
    end
end
if ( unspaced && exist('n','var') && ~isempty(n) && length(info.data)>1 )
    info.data = linspace(info.data(2),info.data(3),n);
end
if (~unspaced && exist('n','var') && ~isempty(n) && length(info.data)~=n)
    warning('XRDMLread:general','different numbers of axis positions and data points')
end
end % readAxisInfo
