"""Visible pipeline test only. Synthetic stage labels are never demonstrations."""
import json
from .adapter import CollectionTeleop


class SmokeTeleop(CollectionTeleop):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,purpose='smoke_test',**kwargs)
        self.smoke_started=False;self.smoke_reset=False;self.smoke_done=False

    def control_frame(self,dt=.02,command_override=None):
        c=self.collector
        if not self.smoke_started:
            self.key('F1');self.smoke_started=True
        if not self.smoke_reset and c.writer is not None and c.writer.count==4:
            self.reset_scene();self.smoke_reset=True;self.key('F1')
        if c.writer is not None:
            frames=c.writer.count-c._stage_start
            duration=105 if c.stage_id==8 else 2
            if frames>=duration:
                if c.stage_id<14:self.key('F2')
                else:
                    self.key('F3');self.smoke_done=True;self.quit=True
                    report={'purpose':'smoke_test','success_demonstration':False,'saved_episode':c.last_episode,
                        'reset_interrupted_episode':True,'stage_sequence':'synthetic_test_labels_only'}
                    ( __import__('pathlib').Path(c.directory)/'smoke_report.json').write_text(json.dumps(report,indent=2))
                    return False
        return super().control_frame(dt,command_override)
