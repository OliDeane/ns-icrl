netconvert --node-files=bidir_3way_junction.nod.xml --edge-files=bidir_3way_junction.edg.xml --output-file=bidir_3way_junction.net.xml
sumo -c bidir_3way_junction.sumocfg
python3 ../parse.py --env=bidir_3way_junction