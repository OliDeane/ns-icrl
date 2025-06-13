netconvert --node-files=sumo_unidir_3way_junction.nod.xml --edge-files=sumo_unidir_3way_junction.edg.xml --output-file=sumo_unidir_3way_junction.net.xml
sumo -c sumo_unidir_3way_junction.sumocfg --seed $1
python3 ../parse.py --env=sumo_unidir_3way_junction --fol=0