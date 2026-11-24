Add-Type -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);' -Name P -Namespace W
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while this process lives (screen may still turn off)
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
while ($true) { Start-Sleep 60 }
