local MACA_ROOT = os.getenv("MACA_PATH") or os.getenv("MACA_HOME") or os.getenv("MACA_ROOT") or "/opt/maca"
local MACA_INCLUDE = path.join(MACA_ROOT, "include")
local CUDA_BRIDGE_INCLUDE = path.join(MACA_ROOT, "tools/cu-bridge/include")
local MXCC = path.join(MACA_ROOT, "mxgpu_llvm/bin/mxcc")

rule("llaisys.maca")
    set_extensions(".maca")

    on_build_file(function (target, sourcefile)
        local objectfile = target:objectfile(sourcefile)
        os.mkdir(path.directory(objectfile))
        local args = {
            "-x", "maca", "-c", sourcefile, "-o", objectfile,
            "-I" .. CUDA_BRIDGE_INCLUDE,
            "-I" .. MACA_INCLUDE,
            "-O3", "-fPIC", "-std=c++17"
        }
        for _, includedir in ipairs(target:get("includedirs")) do
            table.insert(args, "-I" .. includedir)
        end
        for _, define in ipairs(target:get("defines")) do
            table.insert(args, "-D" .. define)
        end
        os.execv(MXCC, args)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

target("llaisys-device-metax")
    set_kind("static")
    add_deps("llaisys-utils")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("../src/device/metax/*.cpp")
    add_includedirs(MACA_INCLUDE)
    add_linkdirs(path.join(MACA_ROOT, "lib"))
    add_links("mcruntime", {public = true})
    add_cxflags("-fPIC")
    on_install(function (target) end)
target_end()
