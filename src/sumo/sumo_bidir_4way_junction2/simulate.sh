netconvert --node-files=sumo_bidir_4way_junction2.nod.xml --edge-files=sumo_bidir_4way_junction2.edg.xml --output-file=sumo_bidir_4way_junction2.net.xml
sumo -c sumo_bidir_4way_junction2.sumocfg -a sumo_bidir_4way_junction2_add.sumocfg --seed $1
python3 ../parse.py --env=sumo_bidir_4way_junction2