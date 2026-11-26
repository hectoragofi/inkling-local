@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set R=C:\Users\hecto\models\inkling
cl /nologo /O2 /EHsc /MD /I %R%\src\llama.cpp-next-ib\ggml\include moebench_lp.cpp /link %R%\build-next-ib\ggml\src\Release\ggml.lib %R%\build-next-ib\ggml\src\Release\ggml-base.lib %R%\build-next-ib\ggml\src\Release\ggml-cpu.lib advapi32.lib
