# Or override any/all of: name, filter, max_works, label_field
./download_openalex.sh ai "primary_topic.subfield.id:1702" 30000 topic
./download_openalex.sh cs "primary_topic.field.id:17"     50000 subfield
./download_openalex.sh general "has_doi:true"            100000 

# Parallelize the download of the general works, which is the largest set
./download_openalex.sh general_part1 "has_doi:true" 50000 part1
./download_openalex.sh general_part2 "has_doi:true" 50000 part2 --offset 50000

# Workers can be used to further parallelize the download of the general works, but be mindful of rate limits and server load
# 1M papers, 4 parallel workers, year ranges 2000-2025 split into 4
./download_openalex.sh huge "has_doi:true" 1000000 field 4

# Or directly via Python with custom year bounds:
python src/data/download.py --dataset openalex --out_dir data/openalex_huge \
  --oa_filter "has_doi:true" --oa_max_works 1000000 --oa_label_field field \
  --oa_workers 4 --oa_year_min 1990 --oa_year_max 2025 \
  --oa_email daachirita@gmail.com