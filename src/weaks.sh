#!/bin/sh

# Questo script richiede 3 parametri per essere eseguito:
# 1 - il nome del programma da eseguire
# 2 - numero di ripetizioni
# 3 - tipo di scaling (1 per prob. crescente, 0 per it. cresccente)
# 3 - grandezza del problema base con un processore
# 4 - numero di iterazioni del problema da eseguire

# Questo script esegue il programma sfruttando OpenMP con un numero 
# di core da 1 al numero di core disponibili sulla macchina
# (estremi inclusi). Il test con p processori viene effettuato su un
# input che ha dimensione N0 * (p^(1/2)), dove N0 e' la dimensione
# dell'input con p=1 thread OpenMP.
#
# Per come è stato implementato il programma parallelo, questo
# significa che all'aumentare del numero p di thread OpenMP, la
# dimensione del problema viene fatta crescere in modo che la quantità
# di lavoro per thread resti costante.


# Ultimo aggiornamento 2023-01-31
# Moreno Marzolla (moreno.marzolla@unibo.it)
# Marco Galeri (marco.galeri@studio.unibo.it)

PROG=$1
REP=$2
TYPE=$3
N0=$4   # base problem size
IT=$5   #iterations



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

for p in `seq $CORES`; do
    echo -n "$p\t"
    # Il comando bc non è in grado di valutare direttamente una radice
    # cubica, che dobbiamo quindi calcolare mediante logaritmo ed
    # esponenziale. L'espressione ($N0 * e(l($p)/2)) calcola
    # $N0*($p^(1/2))
    
    PROB_SIZE=`echo "$N0 * e(l($p)/2)" | bc -l -q`
    IT_SIZE=`echo "$IT * $p" | bc -l -q`

    for rep in `seq $REP`; do
        if [ "$TYPE" = 1 ]; then
        EXEC_TIME="$( OMP_NUM_THREADS=$p "./"$PROG $PROB_SIZE $IT | grep "Elapsed time:" | sed 's/Elapsed time: //' )"
        else
        EXEC_TIME="$( OMP_NUM_THREADS=$p "./"$PROG $N0 $IT_SIZE | grep "Elapsed time:" | sed 's/Elapsed time: //' )"
        fi
        echo -n "${EXEC_TIME}\t"
    done
    echo ""
done
