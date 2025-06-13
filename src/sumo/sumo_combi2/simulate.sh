netconvert --node-files=sumo_combi2.nod.xml --edge-files=sumo_combi2.edg.xml --output-file=sumo_combi2.net.xml
sumo -c sumo_combi2.sumocfg -a sumo_combi2_add.sumocfg --seed $1
python3 ../parse.py --env=sumo_combi2