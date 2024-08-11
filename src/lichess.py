import sys

import aswan
import datazimmer as dz

sys.setrecursionlimit(5000)

main_url = dz.SourceUrl("https://lichess.org/api/tournament")


class GetTournaments(aswan.RequestJsonHandler):
    def parse(self, obj: dict):
        finished = obj["finished"]
        urls = [
            aswan.add_url_params(
                f"{main_url}/{d['id']}/games",
                params={"clocks": True, "accuracy": True},
            )
            for d in finished
        ]
        self.register_links_to_handler(urls, GetTournamentGames)
        return finished


class GetTournamentGames(aswan.RequestHandler):
    process_indefinitely = True
    max_in_parallel = 1

    def parse(self, blob):
        return blob


class LichessDza(dz.DzAswan):
    name: str = "lichess"
    cron: str = "*/30 * * * *"
    starters = {GetTournaments: [main_url]}
