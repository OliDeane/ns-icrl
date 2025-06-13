netconvert --node-files=bidir_4way_junction.nod.xml --edge-files=bidir_4way_junction.edg.xml --output-file=bidir_4way_junction.net.xml
sumo -c bidir_4way_junction.sumocfg -a bidir_4way_junction_add.sumocfg
python3 ../parse.py --env=bidir_4way_junction