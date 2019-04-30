import os
import requests
from bs4 import BeautifulSoup
import re

PAGES = {
	'gameboy': 'https://www.vgmusic.com/music/console/nintendo/gameboy/',
	'n64': 'https://www.vgmusic.com/music/console/nintendo/n64/',
	'gba': 'https://www.vgmusic.com/music/console/nintendo/gba/',
	'ds': 'https://www.vgmusic.com/music/console/nintendo/ds/',
	'3ds': 'https://www.vgmusic.com/music/console/nintendo/3ds/' 
}

BASE_DATA_DIR = '/Users/Siddharth/Desktop/College/Masters/Winter2019/eecs598/MusicGeneration/VariationalAutoEncoder/data/'

def scrape_page_for_midi_href(page, dirname):
	'''
	Assumes that page is a html Object
	Will create a new folder and save all midi files on that page
	'''

	soup = BeautifulSoup(page, 'html.parser')

	# Need to find all pokemon battle songs
	battle_songs = []
	table = soup.find_all('table')[0]
	rows = table.find_all('tr')
	num_rows = len(rows)
	count = 0
	while count < num_rows:
		if is_pokemon_row(rows[count]):
			name = rows[count].contents[1].contents[0].string
			print('collecting songs for game {}'.format(name))
			count += 1
			while count < num_rows and not rows[count].has_attr('class'):
				try:
					a_tag = rows[count].contents[1].contents[0]
				except IndexError as error:
					break
				if 'battle' in a_tag.string.lower():
					print(a_tag['href'])
					battle_songs.append(a_tag['href'])
				count += 1
		count += 1

	return battle_songs

def is_pokemon_row(row):
	if row.has_attr('class'):
		td = row.contents[1]
		name = td.contents[0].string
		if 'pokemon' in name.lower() or 'pokémon' in name.lower():
			print(name)
			return True
	return False

def is_a_midi_file(href):
	return href and re.compile('.mid').search(href)

def main():
	if not os.path.exists(BASE_DATA_DIR): os.mkdir(BASE_DATA_DIR)
	print('Data files will be saved to base dir {}'.format(BASE_DATA_DIR))
	count = 0
	for name, link in PAGES.items():
		print('Collecting midi songs for console {}'.format(name))
		response = requests.get(link, verify=False)
		page = response.content.decode('utf-8')

		battle_songs = scrape_page_for_midi_href(page, BASE_DATA_DIR + name)
		print('####################### {} has {} midi files on this page'.format(name, len(battle_songs)))
		for mfil in battle_songs:
			midiresponse = requests.get(link + mfil, verify=False)
			with open(BASE_DATA_DIR + str(count) + '-' + mfil, 'wb') as f:
				f.write(midiresponse.content)
			count += 1

	print('{} Total pokemon battle songs collected'.format(count))

if __name__ == '__main__':
	main()
