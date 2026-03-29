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

train-bigun data="data/" out="bigun_genome.txt" pop="bigun_pop.pop" pop_size="1000" gens="10000000":
    stdbuf -oL -eL ./build/neat_detector train {{data}} --save {{out}} --save-pop {{pop}} --population {{pop_size}} --generations {{gens}} --save-pop-every 50  2>&1 | stdbuf -oL tee -a bigun_log.txt


clean:
    rm -rf build build-debug
