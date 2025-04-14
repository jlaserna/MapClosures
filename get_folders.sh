#!/bin/bash

while read folder; do
    local_path="./data/HeLiPR/$(echo "$folder" | awk -F'/' '{print $(NF-3)"/"$(NF-2)"/"$(NF-1)"/"$NF}')"

    # Crear la estructura de carpetas localmente
    mkdir -p "$local_path"

    # Descargar solo los archivos del primer nivel, excluyendo subdirectorios
    sftp javier@gala.ieef.upm.es <<EOF
    cd "$folder"
    lcd "$local_path"
    mget *
EOF
done < folders.txt

