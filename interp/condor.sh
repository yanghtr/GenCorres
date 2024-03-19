+Group = "GRAD"
+Project = "GRAPHICS_VISUALIZATION"
+ProjectDescription = "compute correspondence"
+GPUJob = true
Universe     = vanilla
requirements = Eldar
request_GPUs = 1
Executable   = ./interp/batch_interp.sh
Output       = ./interp/log/$(Process).out
Error        = ./interp/log/$(Process).err
Log          = ./interp/log/$(Process).log
arguments = $(Process)
Queue 346

