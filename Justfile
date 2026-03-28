default: build

build:
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build build -j$(nproc)

build-debug:
    cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
    cmake --build build-debug -j$(nproc)

train data="../recordings":
    ./build/neat_detector train {{data}}

train-save data="../recordings" out="best_genome.txt":
    ./build/neat_detector train {{data}} --save {{out}}

eval genome data="../recordings":
    ./build/neat_detector eval {{genome}} {{data}}

infer genome wav:
    ./build/neat_detector infer {{genome}} {{wav}}

clean:
    rm -rf build build-debug
