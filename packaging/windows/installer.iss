; Inno Setup script for Open Broadcast.
; Built by scripts/build.py:  iscc /DAppVersion=1.2.3 /DSourceDir=<dist\OpenBroadcast> /DOutputDir=<dir>

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef AppNumericVersion
  #define AppNumericVersion "0.0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\OpenBroadcast"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist\release"
#endif

#define AppName "Open Broadcast"
#define AppExe "OpenBroadcast.exe"

[Setup]
; Keep this GUID stable forever: it identifies the install for upgrades.
AppId={{6B0E2B3C-6E5D-4D8F-9A51-0F2B7C1E9A42}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Zigdon7
AppPublisherURL=https://github.com/Zigdon-Code-Factory/open-broadcast
AppSupportURL=https://github.com/Zigdon-Code-Factory/open-broadcast/issues
AppUpdatesURL=https://github.com/Zigdon-Code-Factory/open-broadcast/releases
VersionInfoVersion={#AppNumericVersion}
; Per-user install: no UAC prompt, so the in-app updater can run silently.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={localappdata}\Programs\OpenBroadcast
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=OpenBroadcast-{#AppVersion}-windows-x64-setup
SetupIconFile=..\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=force
RestartApplications=no
LicenseFile=..\..\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "autostart"; Description: "Start Open Broadcast when I sign in"; GroupDescription: "Startup:"; Flags: unchecked

[InstallDelete]
; Wipe the previous build's files so stale DLLs never linger across upgrades.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "OpenBroadcast"; ValueData: """{app}\{#AppExe}"" --minimized"; Flags: uninsdeletevalue; Tasks: autostart

[Run]
; Interactive install: offer to launch at the end.
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
; Silent update from the in-app updater (/RELAUNCH=1): relaunch automatically.
Filename: "{app}\{#AppExe}"; Flags: nowait; Check: ShouldRelaunch

[Code]
function ShouldRelaunch: Boolean;
begin
  Result := WizardSilent and (ExpandConstant('{param:RELAUNCH|0}') = '1');
end;
