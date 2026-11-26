from health_utils import berakna_statistik, berakna_sjukdomsandel

class HealthAnalyzer:
    """Klass för analys av hälsodata."""

    def __init__(self, df):
        """Sparar datasetet som ska analyseras."""
        self.df = df

    def berakna_statistik(self, variabler):
        """Beräknar beskrivande statistik för valda variabler."""
        return berakna_statistik(self.df, variabler)

    def berakna_sjukdomsandel(self):
        """Beräknar andelen personer med sjukdom i datasetet."""
        return berakna_sjukdomsandel(self.df)