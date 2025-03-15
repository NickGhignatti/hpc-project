#!/bin/bash

# Questo script richiede 4 parametri per essere eseguito:
# 1 - il nome del programma da eseguire
# 2 - numero di ripetizioni
# 3-4 - La grandezza del problema da eseguire


# Lo script esegue il programma con un numero di core variabile 
# da 1 al numero di core disponibili sulla macchina (estremi inclusi); 
# ogni esecuzione considera sempre la stessa dimensione dell'input, 
# quindi i tempi misurati possono essere usati per calcolare speedup 
# e strong scaling efficiency.
# vengono stampati a video i tempi di esecuzione di tutte le esecuzioni.

# Ultimo aggiornamento 2024-01-31
# Marco Galeri (marco.galeri@studio.unibo.it)
# Moreno Marzolla (moreno.marzolla@unibo.it)

PROG=$1
REP=$2
PROB_SIZE=$3 # problem size 
IT=$4 # problem iterations


if [ ! -f "$PROG" ]; then
    echo
    echo "Non trovo il programma $PROG."
    echo
    exit 1
fi

echo -n "p\t"

for t in `seq $REP`; do
echo -n "t$t\t"
done
echo ""

CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores

if  echo "$PROG" | grep -Eq 'omp' ; then
    TYPE=0;
else
    TYPE=1;
fi

if [ "$TYPE" -eq 0 ]; then
    for p in `seq $CORES`; do
        echo -n "${p}\t"
        for rep in `seq $REP`; do
            EXEC_TIME="$( OMP_NUM_THREADS=$p "./"$PROG $PROB_SIZE $IT  | grep "Elapsed time:" | sed 's/Elapsed time: //' )"
            echo -n "${EXEC_TIME}\t"
        done
        echo ""
    done
else
    for rep in `seq $REP`; do
        EXEC_TIME="$( "./"$PROG $PROB_SIZE $IT | grep "Elapsed time:" | sed 's/Elapsed time: //' )"
        echo -n "${EXEC_TIME}\t"
    done
fi