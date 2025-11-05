$source = "C:\Users\steve\"OneDrive - NOLA Business IT"\source\repos\grocery-ml\dec
$ocrParams = (Get-ChildItem -Path $source -Filter *.png -File | Sort-Object Name | ForEach-Object { "-i `"$($_.FullName)`"" }) -join " "




& C:\Users\steve\Downloads\capture2text_4.6.3\Capture2Text_CLI.exe --line-breaks $ocrParams