# Install/unInstall USER-NEP-GPU package files in LAMMPS
# mode = 0/1/2 for uninstall/install/update

mode=$1

# arg1 = file, arg2 = file it depends on

action () {
    if (test $mode = 0) then
        rm -f ../$1
    elif (! cmp -s $1 ../$1) then
        if (test -z "$2" || test -e ../$2) then
            cp $1 ..
            if (test $mode = 2) then
                echo "  updating src/$1"
            fi
        fi
    elif (test -n "$2") then
        if (test ! -e ../$2) then
            rm -f ../$1
        fi
    fi
}

if (test $mode = 1) then

  # define LAMMPS_VERSION_NUMBER in pair_nep_gpu.cpp
  if (test -e ../version.h) then
    LAMMPS_VERSION=`grep "VERSION" ../version.h | awk -F "[\"\"]" '{print $2}'`
    LAMMPS_VERSION_NUMBER=`date -d "$LAMMPS_VERSION" +%Y%m%d`
    sed -i "s/NUMBER\s\+[0-9]\{8\}/NUMBER $LAMMPS_VERSION_NUMBER/g" pair_nep_gpu.cpp
  fi

  # update Makefile.package: ensure NEP_GPU libs and search paths are present once
  if (test -e ../Makefile.package) then
    # 清理已有的 NEP_GPU / CUDA 相关 token，避免重复
    sed -i -e 's/[^ \t]*nep_gpu[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*cudart[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*cublas[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*cusolver[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*cufft[^ \t]* //g' ../Makefile.package

    # 在 PKG_SYSLIB 行末追加 NEP_GPU 和 CUDA 库
    sed -i -e 's|^PKG_SYSLIB =[ \t]*|&-lnep_gpu -lcudart -lcublas -lcusolver -lcufft |' ../Makefile.package

    # 确保能找到 libnep_gpu.a：假定它放在 LAMMPS src/ 目录（即上一级）
    sed -i -e 's|^PKG_SYSPATH =[ \t]*|&-L.. |' ../Makefile.package

    # 如果环境里有 CUDA_HOME 且存在 lib64 目录，则自动加入 CUDA 搜索路径
    if test "x$CUDA_HOME" != "x" -a -d "$CUDA_HOME/lib64"; then
      sed -i -e "s|^PKG_SYSPATH =[ \t]*|&-L$CUDA_HOME/lib64 |" ../Makefile.package
      echo "  detected CUDA_HOME=$CUDA_HOME, added -L$CUDA_HOME/lib64 to PKG_SYSPATH"
    fi
  fi

elif (test $mode = 0) then

  # uninstall: 从 PKG_SYSLIB / PKG_SYSPATH 行中去掉 NEP_GPU / CUDA 相关设置
  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*nep_gpu[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*cudart[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*cublas[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*cusolver[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*cufft[^ \t]* //g' ../Makefile.package
    sed -i -e 's/[^ \t]*-L\.\.[^ \t]* //g' ../Makefile.package
    if test "x$CUDA_HOME" != "x"; then
      sed -i -e "s|[^ \t]*-L$CUDA_HOME/lib64[^ \t]* ||g" ../Makefile.package
    fi
  fi

fi

# all package files with no dependencies

for file in *.cpp *.h *.cuh; do
    action $file
done

