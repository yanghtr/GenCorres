+Group = "GRAD"
+Project = "GRAPHICS_VISUALIZATION"
+ProjectDescription = "corres"

Universe        = vanilla
requirements  = InMastodon
Executable      = ./batch_icp.sh
Output    = ./log/$(Process).out
Error     = ./log/$(Process).err
Log   = ./log/$(Process).log
arguments = $(Process)
 
Queue 622

