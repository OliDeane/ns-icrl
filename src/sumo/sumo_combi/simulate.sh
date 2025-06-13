netconvert --node-files=sumo_combi.nod.xml --edge-files=sumo_combi.edg.xml --output-file=sumo_combi.net.xml
sumo -c sumo_combi.sumocfg -a sumo_combi_add.sumocfg --seed $1
python3 ../parse.py --env=sumo_combi