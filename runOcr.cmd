

setlocal enabledelayedexpansion
REM set "source=C:\Users\steve\OneDrive - NOLA Business IT\source\repos\grocery-ml\dec"
set "source=./"
set "params="
set "format= --output-format "${capture}${linebreak}${linebreak}${linebreak}####${linebreak}${linebreak}${linebreak} ""
set "outfile= -o ocrtext.txt --output-file-append"
set "tesscfg=  --tess-config-file tess.cfg "
for %%F in ("%source%\*.png") do set "params=!params! -i "%%~fF""

del ocrtext.txt
C:\Users\steve\Downloads\capture2text_4.6.3\Capture2Text_CLI.exe  !tesscfg! !format! !outfile!  --line-breaks  !params!
endlocal


