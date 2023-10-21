# Script to dowlnoad the datasets and models
# Usage:  `bash download.sh --data icl scannet replica`
ROOT_PATH=$(pwd)
DOWNLOAD_ICL=0
DOWNLOAD_SCANNET=0
DOWNLOAD_REPLICA=0

download_dataset() {
    if [ $DOWNLOAD_ICL -eq 1 ]; then
        #############################################################################
        # ICL dataset
        #############################################################################
        cd "$ROOT_PATH"
        icl_folder_path="sample/icl"
        if [ ! -d "$icl_folder_path" ]; then
            # Create the nested folder along with any necessary parent folders
            mkdir -p "$icl_folder_path"
        fi
        cd "$icl_folder_path"

        if [ ! -d "living_room" ]; then
            mkdir "living_room"
        fi
        cd "living_room"

        echo "481.20 0.0 319.5" >> intrinsics.txt
        echo "0.0 -480.0 239.5" >> intrinsics.txt
        echo "0.0 0.0 1.0" >> intrinsics.txt

        if [ ! -d "kt0" ]; then
            echo "[*] Downloading ICL dataset Living Room kt0..."
            mkdir "kt0"
            cd "kt0"
            wget http://www.doc.ic.ac.uk/~ahanda/living_room_traj0_frei_png.tar.gz
            tar -xvzf living_room_traj0_frei_png.tar.gz
            rm living_room_traj0_frei_png.tar.gz
            wget -O livingRoom.gt.sim https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom0n.gt.sim
            cd ..
        fi

        if [ ! -d "kt1" ]; then
            echo "[*] Downloading ICL dataset Living Room kt1..."
            mkdir "kt1"
            cd "kt1"
            wget http://www.doc.ic.ac.uk/~ahanda/living_room_traj1_frei_png.tar.gz
            tar -xvzf living_room_traj1_frei_png.tar.gz
            rm living_room_traj1_frei_png.tar.gz
            wget -O livingRoom.gt.sim https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom1n.gt.sim
            cd ..
        fi

        if [ ! -d "kt2" ]; then
            echo "[*] Downloading ICL dataset Living Room kt2..."
            mkdir "kt2"
            cd "kt2"
            wget http://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz
            tar -xvzf living_room_traj2_frei_png.tar.gz
            rm living_room_traj2_frei_png.tar.gz
            wget -O livingRoom.gt.sim https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom2n.gt.sim
            cd ..
        fi

        if [ ! -d "kt3" ]; then
            echo "[*] Downloading ICL dataset Living Room kt3..."
            mkdir "kt3"
            cd "kt3"
            wget http://www.doc.ic.ac.uk/~ahanda/living_room_traj3_frei_png.tar.gz
            tar -xvzf living_room_traj3_frei_png.tar.gz
            rm living_room_traj3_frei_png.tar.gz
            wget -O livingRoom.gt.sim https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom3n.gt.sim
            cd ..
        fi

    fi
    if [ $DOWNLOAD_SCANNET -eq 1 ]; then
        #############################################################################
        # ScanNet dataset
        #############################################################################
        cd "$ROOT_PATH"
        scannet_folder_path="sample/scannet"
        if [ ! -d "$scannet_folder_path" ]; then
            # Create the nested folder along with any necessary parent folders
            mkdir -p "$scannet_folder_path"
        fi
        echo "[*] If you would like to download the ScanNet data, please fill out an agreement to the ScanNet Terms of Use
             and send it to us at scannet@googlegroups.com."

    fi
    if [ $DOWNLOAD_REPLICA -eq 1 ]; then
        #############################################################################
        # Replica dataset
        #############################################################################

        rename_files() {
            # $1: directory (depth/ or rgb/)
            # $2: filename pattern (depth or frame)
            for file in $2*.$3; do
                # Extract the number part of the filename and remove leading zeros
                num=$(echo $file | grep -o -E '[0-9]+' | sed 's/^0*//')

                # If num is empty, set it to 0
                if [ -z "$num" ]; then
                    num=0
                fi
                # Move and rename the file
                mv "$file" "$1/$num.$3"
            done
        }

        cd "$ROOT_PATH"
        replica_folder_path="sample"
        if [ ! -d "$replica_folder_path" ]; then
            mkdir -p "$replica_folder_path"
        fi
        cd "$replica_folder_path"
        echo "[*] Downloading Replica Dataset from NICE-SLAM..."
        wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
        unzip Replica.zip
        rm Replica.zip
        mv Replica replica

        cd replica
        for dir in office{0..4} room{0..2}; do
            cd "$dir"
            mkdir -p depth rgb
            rename_files "depth" "results/depth" "png"
            rename_files "rgb" "results/frame" "jpg"
            rm -rf results
            cd ..
        done
    fi
}

# Parse command line options
if [ "$1" == "--data" ]; then
    shift  # shift past the --data option
    while (( "$#" )); do
        case "$1" in
            icl)
                DOWNLOAD_ICL=1
                ;;
            scannet)
                DOWNLOAD_SCANNET=1
                ;;
            replica)
                DOWNLOAD_REPLICA=1
                ;;
            *)
                echo "Invalid dataset: $1"
                echo "Usage: $0 --data [icl] [scannet] [replica]"
                exit 1
                ;;
        esac
        shift  # shift past the current dataset name
    done
fi

# Call the download function
download_dataset