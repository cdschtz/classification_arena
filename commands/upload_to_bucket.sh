# uploads data to the GCloud bucket from local/remote machine

# upload (sync) the wikipedia general dataset, ignore macOS DS_Store files
 gsutil -m rsync -x ".*\.DS_Store$" -r $WIKI_SYNTHETIC_PATH $BUCKET_DATA_PATH

# upload (sync) the result files
gsutil -m rsync -x ".*\.DS_Store$" -r ./results/ $BUCKET_RESULTS_PATH
